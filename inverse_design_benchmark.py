"""Headless inverse design benchmark: force projection vs. no refinement on the highest-nu test networks.

Runs the two-style optimization of constraint_projection.ipynb (node displacements and bond stiffnesses, as in
gnn_optimization/cascade_itpo_optimization.ipynb) for a fixed number of epochs, then checks every optimized network
with a LAMMPS compression to 1% strain. Writes one summary row per (network, method), the GNN nu at every epoch, and
the optimized graphs. Plotting lives in constraint_projection.ipynb.

    uv run python inverse_design_benchmark.py --model cascade --out-dir inverse_design_results
    uv run python inverse_design_benchmark.py --model bootstrapped --out-dir inverse_design_results
    uv run python inverse_design_benchmark.py --model residual --checkpoint residual_simulator/residual_mst/checkpoint_epoch_109.pt --tag residual_node

With --model residual (no projection and proj STE only), the bootstrapped rollout uses a residual_simulator_training.py model (history 3) whose harmonic
term takes the rest lengths of the network being optimized.
"""

import argparse
import math
import os
import subprocess
from copy import deepcopy

import pandas as pd
import torch
from torch_geometric.data import Data

import barostat_parameters
import constraint_projection_benchmark as cpb
import lammps_scripts
import network
import residual_simulator_training as rst
from barostat_utils import estimate_initial_box_vel_y_accurate
from graph_utils import compute_angles, get_correct_edge_attr, to_directed_graph
from simulator_residual import ResidualModel
from training_utils import freeze_normalizer
from torch_simulator_wLJ_64 import DifferentiableCompression64
from utils import calc_p_ratio_box_tensor, to_f32, to_f64

DEVICE = cpb.DEVICE
MAX_SIM_LEN = 200
MARGIN = 0.5  # force bound tau = MARGIN * envelope, the best margin in the long-rollout study
P_TARGET = -0.3
ROLLOUT_STEPS = 20
LR = 0.05
D_MIN = 0.3  # fraction of the original bond length
ANG_MIN = 20.0
MAX_MULT, MIN_MULT = 5.0, 1e-6


def reverse_edge_indices(edge_index, num_nodes):
    # edge_index[:, rev_idx[e]] is the reverse of edge_index[:, e]
    src, dst = edge_index
    edge_ids = src * num_nodes + dst
    order = edge_ids.argsort()
    return order[torch.searchsorted(edge_ids[order], dst * num_nodes + src)]


def lammps_box_trajectory(graph, calc_dir, strain=0.01):
    """Compresses the network in LAMMPS like the training data and returns the box at every dump, as graphs with box_tensor only."""
    os.makedirs(calc_dir, exist_ok=True)
    directed = to_directed_graph(graph)
    atoms = [network.Atom(i + 1, 0.0, x, y, 0.0) for i, (x, y) in enumerate(directed.x.tolist())]
    bonds = [network.Bond(atoms[i], atoms[j], length, k) for (i, j), (_, _, length, k) in zip(directed.edge_index.T.tolist(), directed.edge_attr.tolist())]
    lx, ly = graph.box_tensor.tolist()
    box = network.Box(-lx / 2, lx / 2, -ly / 2, ly / 2, -0.1, 0.1)
    network.Network(atoms, bonds, box, network.Header(atoms, bonds, box), masses={1: 1e6}).write_to_file(os.path.join(calc_dir, "network.lmp"))

    compression = lammps_scripts.CompressionSimulation(box_size=lx, strain=strain, overwrite_dump_freq=200, temperature_range=lammps_scripts.TemperatureRange(1e-7, 1e-7, 10.0))
    compression.write_to_file(calc_dir)
    subprocess.run(["lmp", "-in", "in.deformation"], cwd=calc_dir, env=os.environ | {"OMP_NUM_THREADS": "1"}, stdout=subprocess.DEVNULL, check=True)

    with open(os.path.join(calc_dir, "dump.lammpstrj")) as f:
        lines = f.read().splitlines()
    return [
        Data(box_tensor=torch.tensor([float(b[1]) - float(b[0]) for b in (lines[i + 1].split(), lines[i + 2].split())]))
        for i, line in enumerate(lines)
        if line.startswith("ITEM: BOX BOUNDS")
    ]


def optimize(graph, rollout, epochs):
    """Fixed number of Adam epochs on node displacements and bond stiffnesses; returns the last evaluated network."""
    initial_stiff = graph.edge_attr[:, -1]
    min_dist = D_MIN * graph.edge_attr[:, -2]
    rev_idx = reverse_edge_indices(graph.edge_index, graph.num_nodes)

    displacement = torch.nn.Parameter(torch.zeros_like(graph.x))
    stiffness_adjustment = torch.nn.Parameter(torch.zeros_like(initial_stiff))
    optimizer = torch.optim.Adam([displacement, stiffness_adjustment], lr=LR)

    p_history = []
    for _ in range(epochs):
        optimizer.zero_grad()

        # Displace the nodes; the new bond lengths are the new rest lengths
        curr_graph = graph.clone()
        curr_graph.x = graph.x + displacement
        curr_graph.x = curr_graph.x - curr_graph.x.mean(dim=0)
        curr_graph.pos = curr_graph.x
        curr_graph.edge_attr = get_correct_edge_attr(curr_graph, recompute_stiff=True, lj_params=None, panic_at_nontensor_box=True)

        # Scale the stiffnesses symmetrically within [MIN_MULT, MAX_MULT]
        symmetric_adjustment = (stiffness_adjustment + stiffness_adjustment[rev_idx]) / 2
        multiplier = torch.exp(torch.where(symmetric_adjustment >= 0, math.log(MAX_MULT), -math.log(MIN_MULT)) * torch.tanh(symmetric_adjustment))
        curr_graph.edge_attr = torch.column_stack((curr_graph.edge_attr[:, :3], initial_stiff * multiplier))

        curr_traj = rollout(curr_graph)
        new_p = calc_p_ratio_box_tensor(curr_traj)

        distance_loss = torch.sum(torch.relu(min_dist - curr_graph.edge_attr[:, -2]) ** 2)
        angle_loss = torch.sum(torch.nn.functional.softplus(ANG_MIN - compute_angles(curr_graph.x, graph.angle_index, curr_graph.box_tensor), beta=3.0))
        p_loss = (new_p - P_TARGET) ** 2
        loss = distance_loss + angle_loss + p_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([displacement, stiffness_adjustment], max_norm=0.5)
        optimizer.step()
        p_history.append(new_p.item())
    return {"graph": curr_graph.detach(), "traj": [g.detach() for g in curr_traj], "p": p_history}


def bootstrapped_rollout(g, model, refine, config, history, rollout_steps):
    """The differentiable MD bootstrap of utils.specialized_rollout_STE (no Langevin noise), then constrained_rollout."""
    dump_period = config["default_skip"]
    md = DifferentiableCompression64(g.num_nodes, temp_langevin=0.0).run_simulator(to_f64(g.clone()), dump_period * history + 1, r0=g.edge_attr[:, -2], device=DEVICE)[0]
    frames = [to_f32(md[i * dump_period]) for i in range(history + 1)]
    for frame in frames:
        frame.pos = frame.x
    box_vel_y = estimate_initial_box_vel_y_accurate(*frames[-3:], dump_period * config["dt"])
    box_delta_x = frames[-1].box_tensor[0] - frames[-2].box_tensor[0]
    return cpb.constrained_rollout([model] * (history + 1), frames, rollout_steps, config, box_delta_x, box_vel_y, g.edge_attr[:, -2], refine, grad=True)


def set_r0(model, r0):
    """The residual model's harmonic term uses the rest lengths of the network being optimized."""
    model.r0 = r0
    return model


def load_residual(checkpoint):
    config = barostat_parameters.node_optimizated
    init_graph = Data(x=torch.ones((100, cpb.HISTORY * 2)), edge_attr=torch.ones((100, 4)))
    model = ResidualModel(init_graph, 128, 2, 3, rst.MASS, rst.DUMP_PERIOD * config["dt"]).to(DEVICE)
    model.load_checkpoint(checkpoint)
    model = freeze_normalizer(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cascade", "bootstrapped", "residual"], required=True)
    parser.add_argument("--checkpoint", help="residual model checkpoint, required with --model residual")
    parser.add_argument("--tag", help="output directory name and `model` column, by default the model")
    parser.add_argument("--out-dir", default="inverse_design_results")
    parser.add_argument("--n-networks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-test-sims", type=int, default=60)
    parser.add_argument("--n-train-sims", type=int, default=100)
    args = parser.parse_args()
    if args.model == "residual" and args.checkpoint is None:
        parser.error("--model residual needs --checkpoint")
    tag = args.tag or args.model

    train_data, test_data = cpb.load_data(args.n_train_sims, args.n_test_sims, MAX_SIM_LEN)
    tau = MARGIN * cpb.force_envelope(train_data)
    refines = {"default": cpb.no_refine, "proj STE": cpb.ste(cpb.make_refine_force(tau)), "proj IFT": cpb.make_refine_force_ift(tau)}
    if args.model == "residual":
        del refines["proj IFT"]  # STE only: IFT doubles the run time
    networks = sorted(range(len(test_data)), key=lambda i: calc_p_ratio_box_tensor(test_data[i]).item(), reverse=True)[: args.n_networks]

    if args.model == "cascade":
        cascade = cpb.load_cascade(test_data[0])
    elif args.model == "bootstrapped":
        gnn_simulator = cpb.load_bootstrapped()
    else:
        gnn_simulator = load_residual(args.checkpoint)

    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    rows, history_rows, graphs = [], [], {}
    for net in networks:
        sim = test_data[net]
        graph = sim[0].clone().to(DEVICE)
        if args.model == "cascade":
            config = barostat_parameters.node_optimizated
            delta_x = (sim[4].box_tensor[0] - sim[3].box_tensor[0]).item()
            make_rollout = lambda refine: lambda g: cpb.constrained_rollout(cascade, [g], ROLLOUT_STEPS, config, delta_x, 0.0, g.edge_attr[:, -2], refine, grad=True)
            last_frame = ROLLOUT_STEPS
        else:
            # Compute dumping period (N MD steps in 1 dump step)
            sim_strain = (sim[1].box.x - sim[-1].box.x) / sim[0].box.x
            config = deepcopy(barostat_parameters.node_optimizated)
            config["default_skip"] = int(int(sim_strain / 1e-5 / 0.01) / len(sim)) + 1
            make_rollout = lambda refine: lambda g: bootstrapped_rollout(g, gnn_simulator, refine, config, cpb.HISTORY, ROLLOUT_STEPS)
            if args.model == "residual":  # same dump period estimate as residual_simulator_training.evaluate_rollouts
                make_rollout = lambda refine: lambda g: bootstrapped_rollout(g, set_r0(gnn_simulator, g.edge_attr[:, -2]), refine, config, cpb.HISTORY, ROLLOUT_STEPS)
            last_frame = cpb.HISTORY + ROLLOUT_STEPS

        net_dir = os.path.join(out_dir, "lammps", f"net{net}")
        original = lammps_box_trajectory(graph, os.path.join(net_dir, "original"))
        for method, refine in refines.items():
            result, elapsed = cpb.timed(lambda: optimize(graph, make_rollout(refine), args.epochs))
            lammps = lammps_box_trajectory(result["graph"], os.path.join(net_dir, method.replace(" ", "_")))
            rows.append({
                "model": tag, "network": net, "method": method, "epochs": args.epochs, "time": elapsed,
                "gt_p": calc_p_ratio_box_tensor(sim).item(),
                "lammps_p_original": calc_p_ratio_box_tensor(original).item(),
                "gnn_p": result["p"][-1],
                "lammps_p_last_frame": calc_p_ratio_box_tensor(lammps, last_frame).item(),
                "lammps_p": calc_p_ratio_box_tensor(lammps).item(),
            })
            history_rows += [{"model": tag, "network": net, "method": method, "epoch": i, "gnn_p": p} for i, p in enumerate(result["p"])]
            graphs[(net, method)] = result["graph"].cpu()
            print(f"net {net:>2} | {method:<8} | GNN nu {result['p'][-1]:+.3f} | LAMMPS nu {rows[-1]['lammps_p']:+.3f} (original {rows[-1]['lammps_p_original']:+.3f}) | {elapsed:.0f} s", flush=True)

            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
            pd.DataFrame(history_rows).to_csv(os.path.join(out_dir, "history.csv"), index=False)
            torch.save(graphs, os.path.join(out_dir, "graphs.pt"))


if __name__ == "__main__":
    main()
