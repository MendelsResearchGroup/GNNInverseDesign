"""Headless benchmark: constraint projection vs. ITPO on long rollouts.

Runs every method on the test sims for one model (simulator cascade or bootstrapped GNN simulator) and
writes one CSV row per (method, margin, sim, checkpoint step). Plotting lives in constraint_projection.ipynb.

    uv run python constraint_projection_benchmark.py --model cascade --out-dir constraint_projection_results
    uv run python constraint_projection_benchmark.py --model bootstrapped --out-dir constraint_projection_results
"""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS
import time
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
import torchopt
from torch_geometric.data import Data
from tqdm import tqdm

import barostat_parameters
import itpo_weights
from barostat_utils import estimate_initial_box_vel_y_accurate, update_box_y_thermodynamic
from graph_utils import get_correct_edge_attr, prepare_traj
from ift import specialized_rollout_implicit
from itpo_weights import DatasetType, ITPOWeights, ModelType
from pressure import compute_per_particle_forces
from simulator_SA_cpu_test import Model as VelocityModel
from training_utils import freeze_normalizer
from utils import (
    build_velocity_graph_correction,
    calc_p_ratio_box_tensor,
    compute_combined_physics_loss,
    load_and_split_dataset,
    rollout_cascade,
    simulate_then_rollout,
    specialized_rollout_cascade,
)

DEVICE = "cuda"
DATASET_TYPE = DatasetType.NodeOptimized
POISSON_THRESHOLD = 0.1  # models were trained on nu >= 0.1 only
HISTORY = 3  # bootstrapped GNN simulator history
MARGINS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0]
SEED = 0
COMBINED_MARGIN = 1.0

# Cascade NodeOptimized weights as committed in HEAD (the working-copy entry comes from a 5-sim search and barely refines)
CASCADE_ITPO_WEIGHTS = ITPOWeights(50, 2.4347939215508626e-06, 0.2571693753028027, 3.0716147827138016e-07, 0.000396262687255812, 1e-8, 1e-8)


def next_graph(a, curr_graph, prev_graph, box_delta_x, r0, box_vel_y, barostat_config):
    """Forward Euler + uniaxial compression + barostat, the same sequence as the ITPO closure in physical_inference_step."""
    num_particles = curr_graph.num_nodes
    stride_dt = barostat_config["default_skip"] * barostat_config["dt"]

    x_next = curr_graph.pos + (curr_graph.pos - prev_graph.pos) + a
    predicted_graph = Data(x=x_next, pos=x_next, edge_index=curr_graph.edge_index, edge_attr=curr_graph.edge_attr, box_tensor=curr_graph.box_tensor)

    compressed_ly, new_vel_y = update_box_y_thermodynamic(
        positions=x_next,
        edge_index=curr_graph.edge_index,
        edge_attr=curr_graph.edge_attr,
        current_box=curr_graph.box_tensor,
        r0=r0.float(),
        box_vel_y=box_vel_y,
        W_y=barostat_config["C_coupling"] * num_particles * (stride_dt**2),
        damping=barostat_config["damping"] * num_particles * stride_dt,
        stride_dt=stride_dt,
        target_pressure=barostat_config["target_pressure"],
        temperature=barostat_config["temperature"],
    )
    predicted_graph.box_tensor = torch.stack([curr_graph.box_tensor[0] + box_delta_x, compressed_ly])
    predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=None, panic_at_nontensor_box=True)

    return predicted_graph, new_vel_y


def constrained_rollout(models, rollout, num_steps, barostat_config, box_delta_x, box_vel_y, r0, refine, grad=False):
    """rollout_cascade with a refine(a, make_graph, r0, step) hook between the GNN and the integrator.

    A single bootstrapped simulator goes through the same loop as [model] * (history + 1).
    grad=True keeps the autograd graph through the whole rollout (pair with ste(refine)).
    """
    rollout = list(rollout)
    for _ in range(num_steps):
        active_model = models[min(len(rollout), len(models)) - 1]
        input_graph = build_velocity_graph_correction(rollout[-len(models):], panic_at_positions=False)
        curr_graph = rollout[-1]
        prev_graph = rollout[-2] if len(rollout) > 1 else rollout[-1]
        make_graph = partial(next_graph, curr_graph=curr_graph, prev_graph=prev_graph, box_delta_x=box_delta_x, r0=r0, box_vel_y=box_vel_y, barostat_config=barostat_config)

        with torch.set_grad_enabled(grad):
            a = active_model.output_normalizer.inverse(active_model(input_graph, is_training=False))
        a = refine(a, make_graph, r0, len(rollout))

        with torch.set_grad_enabled(grad):
            predicted_graph, box_vel_y = make_graph(a)
        rollout.append(predicted_graph)
    return rollout


def no_refine(a, make_graph, r0, step):
    return a


def bond_stiffness_diag(graph):
    # Longitudinal part of the harmonic bond Hessian, summed per particle: H_ii = sum_j 2 k_ij n_ij n_ij^T
    src, dst = graph.edge_index
    vec = graph.pos[dst] - graph.pos[src]
    vec = vec - torch.round(vec / graph.box_tensor) * graph.box_tensor
    n = vec / vec.norm(dim=1, keepdim=True)
    blocks = 2 * graph.edge_attr[:, -1, None, None] * n[:, :, None] * n[:, None, :]
    return torch.zeros(graph.num_nodes, 2, 2, device=blocks.device).index_add_(0, src, blocks)


def make_refine_force(tau, sweeps=5, omega=0.5):
    # Jacobi projection onto |F_i| <= tau: F(x + d) ~ F - H d, so d_i = H_ii^-1 r_i removes the out-of-band residual r
    def refine(a, make_graph, r0, step):
        with torch.no_grad():
            for _ in range(sweeps):
                predicted_graph, _ = make_graph(a)
                forces = compute_per_particle_forces(predicted_graph, r0=r0, cutoff=None)
                residual = forces - forces.clamp(-tau[step], tau[step])
                a = a + omega * (torch.linalg.pinv(bond_stiffness_diag(predicted_graph)) @ residual[:, :, None])[:, :, 0]
        return a
    return refine


class ProjectionIFT(torch.autograd.Function):
    """Passes the projected acceleration through; the backward is the implicit derivative of the min-norm projection
    onto the active force constraints, d a*/d a_nn = I - J^T (J J^T)^-1 J with J = dF_active/da.

    As in ift.ImplicitPhysicsRefinement, only the a_nn path is differentiated and the solve is 10 CG iterations.
    """

    @staticmethod
    def forward(ctx, a_nn, a, forces, tau):
        ctx.save_for_backward(a)
        ctx.forces, ctx.tau = forces, tau
        return a.clone()

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        with torch.enable_grad():
            a = a.detach().requires_grad_(True)
            f = ctx.forces(a)
            active = (f.abs() > ctx.tau).to(f.dtype)
            if not active.any():
                return grad, None, None, None
            probe = torch.zeros_like(f, requires_grad=True)
            probe_vjp = torch.autograd.grad(f, a, probe, create_graph=True)[0]  # J^T probe, linear in probe
            jt = lambda w: torch.autograd.grad(f, a, active * w, retain_graph=True)[0]
            j = lambda u: active * torch.autograd.grad(probe_vjp, probe, u, retain_graph=True)[0]
            w = torchopt.linear_solve.solve_cg(maxiter=10)(matvec=lambda w: j(jt(w)), b=j(grad))
            return grad - jt(w), None, None, None


def make_refine_force_ift(tau, sweeps=5, omega=0.5):
    project = make_refine_force(tau, sweeps, omega)

    def refine(a_nn, make_graph, r0, step):
        forces = lambda a: compute_per_particle_forces(make_graph(a)[0], r0=r0, cutoff=None)
        return ProjectionIFT.apply(a_nn, project(a_nn.detach(), make_graph, r0, step), forces, tau[step])
    return refine


def make_refine_itpo(weights):
    # Same Adam refinement as physical_inference_step
    def refine(a_nn, make_graph, r0, step):
        a = torch.nn.Parameter(a_nn.clone())
        optimizer = torch.optim.Adam([a], lr=weights.learning_rate)
        for _ in range(weights.refinement_iterations):
            with torch.enable_grad():
                optimizer.zero_grad()
                predicted_graph, _ = make_graph(a)
                energy_loss, force_loss, pressure_loss = compute_combined_physics_loss(predicted_graph, r0=r0, lj_cutoff=None, target_Pyy=0.0)
                loss = (
                    torch.mean((a - a_nn) ** 2)
                    + weights.lambda_energy * energy_loss
                    + weights.lambda_force * force_loss
                    + weights.lambda_pressure * pressure_loss
                )
                loss.backward()
            optimizer.step()
        return a.detach()
    return refine


def ste(refine):
    # Straight-through: the forward pass uses the refined acceleration, the backward pass treats refine as identity
    return lambda a, make_graph, r0, step: a + (refine(a.detach(), make_graph, r0, step) - a).detach()


def chain(*refines):
    def refine(a, make_graph, r0, step):
        for f in refines:
            a = f(a, make_graph, r0, step)
        return a
    return refine


def load_sims(files, max_sim_len):
    return [prepare_traj(torch.load(file, weights_only=False)[:max_sim_len], None, calc_angles=False) for file in tqdm(files, desc="loading")]


def split_files():
    poisson_buckets = [
        {"max": 0.1, "count": 300},             # P < 0.1
        {"min": 0.1, "max": 0.2, "count": 100}, # 0.1 <= P < 0.2
        {"min": 0.2, "count": 100}              # P >= 0.2
    ]
    return load_and_split_dataset(
        registry_path="./data/data_registry.csv",
        target_data_type=DATASET_TYPE,
        possion_buckets=poisson_buckets,
        split_ratios=(0.5, 0.25, 0.25),
        seed=42,
    )


def load_data(n_train_sims, n_test_sims, max_sim_len):
    """Train sims with nu >= POISSON_THRESHOLD (used only for the force envelope) and the first test sims."""
    train_files, _, test_files = split_files()
    train_data = [sim for sim in load_sims(train_files, max_sim_len) if calc_p_ratio_box_tensor(sim) >= POISSON_THRESHOLD][:n_train_sims]
    return train_data, load_sims(test_files[:n_test_sims], max_sim_len)


def force_envelope(sims):
    # Per-frame max |F_i| over the sims; the bound is tau = margin * envelope
    return torch.stack([
        torch.stack([compute_per_particle_forces(g, r0=sim[0].edge_attr[:, -2], cutoff=None).abs().max() for g in sim])
        for sim in sims
    ]).max(dim=0).values.to(DEVICE)


def load_cascade(example_sim):
    model_save_path = os.path.join("./trained_models", f"{DATASET_TYPE}", "cascade", "refined")
    cascade = []
    for h in range(len(os.listdir(model_save_path))):
        init_graph = build_velocity_graph_correction([example_sim[i].cpu() for i in range(max(1, h + 1))], panic_at_positions=False).to(DEVICE)
        model = VelocityModel(init_graph, 128, 2, 3).to(DEVICE)
        model.load_checkpoint(os.path.join(model_save_path, f"model_refined_h{h}_nl2_mlp3_epochs100.pt"))
        model = freeze_normalizer(model)
        for param in model.parameters():
            param.requires_grad = False
        cascade.append(model)
    return cascade


def load_bootstrapped():
    init_graph = Data(x=torch.ones((100, HISTORY * 2)), edge_attr=torch.ones((100, 4)))
    model = VelocityModel(init_graph, 128, 2, 3).to(DEVICE)
    model.load_checkpoint(f"./new_trained_models/{DATASET_TYPE}/MST/checkpoint_epoch_120.pt")
    model = freeze_normalizer(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


def evaluate(rollout, sim, offset, checkpoints, envelope):
    """One row per checkpoint: metrics after `step` GNN steps, i.e. at frame offset + step."""
    rollout = [g.to(DEVICE) for g in rollout[: offset + checkpoints[-1] + 1]]
    r0 = sim[0].edge_attr[:, -2].to(DEVICE)
    force_ratio = np.array([(compute_per_particle_forces(g, r0=r0, cutoff=None).abs().max() / envelope[i]).item() for i, g in enumerate(rollout)])
    rows = []
    for step in checkpoints:
        frame = offset + step
        rows.append({
            "step": step,
            "gt_p": calc_p_ratio_box_tensor(sim[: frame + 1]).item(),
            "pred_p": calc_p_ratio_box_tensor(rollout[: frame + 1]).item(),
            "mse": torch.nn.functional.mse_loss(rollout[frame].pos.cpu(), sim[frame].pos.cpu()).item(),
            "max_force_ratio": force_ratio[1 : frame + 1].max(),
        })
    return rows


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main():
    # Scatter-add backward is nondeterministic on CUDA, which 50 Adam iterations of ITPO amplify into ~1e-3 in nu
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cascade", "bootstrapped"], required=True)
    parser.add_argument("--out-dir", default="constraint_projection_results")
    parser.add_argument("--n-test-sims", type=int, default=60)
    parser.add_argument("--n-train-sims", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=200)
    args = parser.parse_args()

    checkpoints = list(range(50, args.rollout_steps + 1, 50))
    max_sim_len = args.rollout_steps + HISTORY + 1

    train_data, test_data = load_data(args.n_train_sims, args.n_test_sims, max_sim_len)
    envelope = force_envelope(train_data)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}.csv")
    rows = []

    def record(method, margin, sim_idx, rollout, sim, offset, elapsed):
        for row in evaluate(rollout, sim, offset, checkpoints, envelope):
            rows.append({"model": args.model, "method": method, "margin": margin, "sim": sim_idx, "time": elapsed, **row})

    if args.model == "cascade":
        cascade = load_cascade(test_data[0])
        barostat_config = barostat_parameters.node_optimizated
        mean_delta_x = np.mean([(sim[4].box_tensor[0] - sim[3].box_tensor[0]).item() for sim in test_data])
        current_weights = itpo_weights.get_params(ModelType.SimulatorCascade, DATASET_TYPE)

        for sim_idx, sim in enumerate(tqdm(test_data, desc="cascade")):
            torch.manual_seed(SEED + sim_idx)
            r0 = sim[0].edge_attr[:, -2].to(DEVICE)
            start = [sim[0].to(DEVICE)]
            run = lambda refine: constrained_rollout(cascade, start, args.rollout_steps, barostat_config, mean_delta_x, 0.0, r0, refine)

            rollout, elapsed = timed(lambda: rollout_cascade(cascade, sim[0].cpu().detach(), args.rollout_steps, barostat_config, None, mean_delta_x, DEVICE))
            record("default", np.nan, sim_idx, rollout, sim, 0, elapsed)
            rollout, elapsed = timed(lambda: specialized_rollout_cascade(sim[0].cpu().detach(), cascade, barostat_config, mean_delta_x, CASCADE_ITPO_WEIGHTS, args.rollout_steps, DEVICE))
            record("ITPO", np.nan, sim_idx, rollout, sim, 0, elapsed)
            rollout, elapsed = timed(lambda: specialized_rollout_cascade(sim[0].cpu().detach(), cascade, barostat_config, mean_delta_x, current_weights, args.rollout_steps, DEVICE))
            record("ITPO (current weights)", np.nan, sim_idx, rollout, sim, 0, elapsed)
            for margin in MARGINS:
                rollout, elapsed = timed(lambda: run(make_refine_force(margin * envelope)))
                record("proj force", margin, sim_idx, rollout, sim, 0, elapsed)
            rollout, elapsed = timed(lambda: run(chain(make_refine_itpo(CASCADE_ITPO_WEIGHTS), make_refine_force(COMBINED_MARGIN * envelope))))
            record("ITPO + proj force", COMBINED_MARGIN, sim_idx, rollout, sim, 0, elapsed)

            pd.DataFrame(rows).to_csv(out_path, index=False)

    else:
        gnn_simulator = load_bootstrapped()
        weights = itpo_weights.get_params(ModelType.GNNModel, DATASET_TYPE)

        for sim_idx, sim in enumerate(tqdm(test_data, desc="bootstrapped")):
            # The MD bootstrap adds Langevin noise
            torch.manual_seed(SEED + sim_idx)
            r0 = sim[0].edge_attr[:, -2].to(DEVICE)

            # Compute dumping period (N MD steps in 1 dump step)
            sim_strain = (sim[1].box.x - sim[-1].box.x) / sim[0].box.x
            dump_period = int(int(sim_strain / 1e-5 / 0.01) / len(sim)) + 1
            barostat_config = deepcopy(barostat_parameters.node_optimizated)
            barostat_config["default_skip"] = dump_period
            md_steps = dump_period * HISTORY + 1

            # ITPO uses its own MD bootstrap; everything else shares one bootstrap per sim
            rollout, elapsed = timed(lambda: specialized_rollout_implicit(sim[0].cpu().detach(), gnn_simulator, HISTORY, barostat_config, None, weights, md_steps, args.rollout_steps, DEVICE))
            record("ITPO", np.nan, sim_idx, rollout, sim, HISTORY, elapsed)

            bootstrap, bootstrap_time = timed(lambda: [g.to(DEVICE) for g in simulate_then_rollout(sim[0].cpu().detach(), gnn_simulator, HISTORY, barostat_config, None, md_steps, 0, DEVICE)[: HISTORY + 1]])
            box_delta_x = bootstrap[-1].box_tensor[0] - bootstrap[-2].box_tensor[0]
            box_vel_y = estimate_initial_box_vel_y_accurate(bootstrap[-3], bootstrap[-2], bootstrap[-1], dump_period * barostat_config["dt"])
            run = lambda refine: constrained_rollout([gnn_simulator] * (HISTORY + 1), bootstrap, args.rollout_steps, barostat_config, box_delta_x, box_vel_y, r0, refine)

            rollout, elapsed = timed(lambda: run(no_refine))
            record("default", np.nan, sim_idx, rollout, sim, HISTORY, elapsed + bootstrap_time)
            for margin in MARGINS:
                rollout, elapsed = timed(lambda: run(make_refine_force(margin * envelope)))
                record("proj force", margin, sim_idx, rollout, sim, HISTORY, elapsed + bootstrap_time)
            rollout, elapsed = timed(lambda: run(chain(make_refine_itpo(weights), make_refine_force(COMBINED_MARGIN * envelope))))
            record("ITPO + proj force", COMBINED_MARGIN, sim_idx, rollout, sim, HISTORY, elapsed + bootstrap_time)

            pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
