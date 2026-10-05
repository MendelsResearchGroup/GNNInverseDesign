"""Training of the bootstrapped GNN simulator, plain or as a correction to the harmonic forces.

The training loops are the one-step (OST) and multi-step (MST) loops of gnn_simulator_training.ipynb, on the training
sims with nu >= 0.1 of a dataset (split as in constraint_projection_benchmark.split_files). After training, every test
sim is rolled out from its first HISTORY + 1 ground-truth frames without refinement and scored with
constraint_projection_benchmark.evaluate; MST runs are also scored every 10 epochs. With --training none, an existing
checkpoint (or, with --model harmonic, the harmonic step alone) is only scored.

    uv run python residual_simulator_training.py --model residual --training ost --out-dir residual_simulator
    uv run python residual_simulator_training.py --model residual --training mst --out-dir residual_simulator
    uv run python residual_simulator_training.py --dataset lj_noisy --model plain --training none --checkpoint PATH --tag reference --out-dir residual_simulator/lj_noisy
"""

import argparse
import os
import random
import time
import types
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import barostat_parameters
import constraint_projection_benchmark as cpb
from barostat_utils import estimate_initial_box_vel_y_accurate, update_box_y_thermodynamic
from graph_utils import LJInteractionParams, get_correct_edge_attr
from itpo_weights import DatasetType
from simulator_residual import ResidualModel, harmonic_acceleration
from simulator_SA_cpu_test import Model as VelocityModel
from training_utils import ModelInputs, huber_loss
from utils import build_velocity_graph_correction, calc_p_ratio_box_tensor

DEVICE = cpb.DEVICE
HISTORY = cpb.HISTORY
MAX_SIM_LEN = 200
MASS = 1e6 * 1.0364269e-4  # particle mass of the training data times mvv2e of LAMMPS metal units, as in torch_simulator_wLJ_64
DUMP_PERIOD = 200  # MD steps per frame in every dataset

DATASETS = {
    "node_optimized": {"registry": "./data/data_registry.csv", "data_type": DatasetType.NodeOptimized, "barostat": barostat_parameters.node_optimizated},
    "stiff_optimized": {"registry": "./data/data_registry.csv", "data_type": DatasetType.StiffOptimized, "barostat": barostat_parameters.stiff_optimized},
    "lj_noisy": {
        "registry": "./data/data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/data_registry.csv",
        "data_type": DatasetType.LJNoisy,
        "barostat": barostat_parameters.lj_noisy,
        "lj_params": LJInteractionParams(epsilon=0.01, sigma=1.0, cutoff=1.122),
        "poisson_buckets": [{"min": 0.0, "max": 1.0, "count": 400}],  # no nu < 0.1 sims, full range as in gnn_simulator_training.ipynb
    },
    "noisy": {"directory": "./data/noisy_dump200", "barostat": barostat_parameters.noisy},  # no registry
}

# OST and MST settings of gnn_simulator_training.ipynb
EPOCHS = 100
MST_EPOCHS = 150
MAX_ROLLOUT_STEPS = 10
FREEZE_NORM_EPOCH = 5
VAL_SIMS = 20
TRAIN_LIMIT = 15
ACCUMULATION_STEPS = 10
LEARNING_RATE = 1e-3
GAMMA = 0.995


def one_step_samples(sim):
    """(input graph, model inputs) for the first TRAIN_LIMIT one-step targets of a sim, as in the OST loop."""
    for idx in range(TRAIN_LIMIT):
        input_graphs_raw = [sim[k].detach().cpu() for k in range(idx, idx + HISTORY + 1)]
        input_graph = build_velocity_graph_correction(input_graphs_raw, panic_at_positions=False).to(DEVICE)
        model_inputs = ModelInputs(input_graphs_raw[-2].to(DEVICE), input_graphs_raw[-1].to(DEVICE), sim[idx + HISTORY + 1].to(DEVICE))
        yield input_graph, model_inputs


def dataset_files(dataset):
    """Train, val and test files: the benchmark split for registry datasets, the same ratios over the files otherwise."""
    if "registry" in dataset:
        return cpb.split_files(dataset["registry"], dataset["data_type"], dataset.get("poisson_buckets"))
    files = sorted(os.path.join(dataset["directory"], f) for f in os.listdir(dataset["directory"]))
    random.Random(cpb.SEED).shuffle(files)
    n_train, n_val = len(files) // 2, len(files) // 4
    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def load_dataset(dataset, n_train_sims, n_test_sims):
    """Train sims with nu >= POISSON_THRESHOLD, as in constraint_projection_benchmark.load_data, plus val and test sims."""
    train_files, val_files, test_files = dataset_files(dataset)
    lj_params = dataset.get("lj_params")
    train_data = [sim for sim in cpb.load_sims(train_files, MAX_SIM_LEN, lj_params) if calc_p_ratio_box_tensor(sim) >= cpb.POISSON_THRESHOLD][:n_train_sims]
    return train_data, cpb.load_sims(val_files[:VAL_SIMS], MAX_SIM_LEN, lj_params), cpb.load_sims(test_files[:n_test_sims], MAX_SIM_LEN, lj_params)


class HarmonicOnly(torch.nn.Module):
    """a = a_harm: the residual model's physics term without the network."""

    output_normalizer = types.SimpleNamespace(inverse=lambda a: a)

    def __init__(self, stride_dt, cutoff):
        super().__init__()
        self.stride_dt, self.cutoff, self.r0 = stride_dt, cutoff, None

    def forward(self, data, is_training=False):
        return harmonic_acceleration(data, self.r0, MASS, self.stride_dt, self.cutoff)


def harmonic_r2(sims, stride_dt, cutoff):
    """Fraction of the ground-truth one-step acceleration variance explained by the harmonic step alone."""
    targets, harmonic = [], []
    for sim in sims:
        r0 = sim[0].edge_attr[:, -2].to(DEVICE)
        for input_graph, inputs in one_step_samples(sim):
            targets.append((inputs.target_position - inputs.cur_position) - (inputs.cur_position - inputs.prev_position))
            harmonic.append(harmonic_acceleration(input_graph, r0, MASS, stride_dt, cutoff))
    targets, harmonic = torch.cat(targets), torch.cat(harmonic)
    return (1 - (targets - harmonic).pow(2).sum() / (targets - targets.mean(dim=0)).pow(2).sum()).item()


def train(model, training_data, val_data, model_save_directory, shuffle=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, last_epoch=-1)

    log = []
    optimizer.zero_grad()
    for epoch in range(EPOCHS):
        t_start = time.perf_counter()

        if epoch == FREEZE_NORM_EPOCH:
            model.node_normalizer.frozen = True
            model.edge_normalizer.frozen = True
            model.output_normalizer.frozen = True

        total_acc_loss = 0
        total_val_loss = 0
        total_val_pos_mse = 0
        train_samples = 0
        val_samples = 0

        model.train()
        order = list(training_data)
        if shuffle:
            random.Random(cpb.SEED + epoch).shuffle(order)
        for sim in order:
            model.r0 = sim[0].edge_attr[:, -2].to(DEVICE)
            for i, (input_graph, model_inputs) in enumerate(one_step_samples(sim)):
                # Forward and Loss
                model_output = model(input_graph, is_training=True)
                acc_loss = huber_loss(model, model_output, model_inputs, is_training=True)

                # Backward
                loss_for_backward = acc_loss / ACCUMULATION_STEPS
                loss_for_backward.backward()

                # Optimization
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_acc_loss += acc_loss.item()
                train_samples += 1

            optimizer.step()
            optimizer.zero_grad()

        # Validation
        with torch.no_grad():
            model.eval()
            for val_sim in val_data:
                model.r0 = val_sim[0].edge_attr[:, -2].to(DEVICE)
                for input_graph, val_inputs in one_step_samples(val_sim):
                    model_output = model(input_graph, is_training=False)
                    val_loss = huber_loss(model, model_output, val_inputs, is_training=False)

                    # Update to next state and check position MSE
                    pred_graph = model.update(val_inputs, model_output)
                    pos_mse = torch.nn.functional.mse_loss(pred_graph.pos, val_inputs.target_graph.pos)

                    total_val_loss += val_loss.item()
                    total_val_pos_mse += pos_mse.item()
                    val_samples += 1

        lr_scheduler.step()

        model.save_checkpoint(os.path.join(model_save_directory, f"checkpoint_epoch_{epoch}.pt"))
        log.append({"epoch": epoch, "train_loss": total_acc_loss / train_samples, "val_loss": total_val_loss / val_samples,
                    "val_pos_mse": total_val_pos_mse / val_samples, "time": time.perf_counter() - t_start})
        pd.DataFrame(log).to_csv(os.path.join(model_save_directory, "train_log.csv"), index=False)
        print(f"Epoch {epoch + 1:>3} | Train Loss: {log[-1]['train_loss']:.3e} | Val Loss: {log[-1]['val_loss']:.3e} | "
              f"Val Pos MSE: {log[-1]['val_pos_mse']:.3e} | Time: {log[-1]['time']:.2f} s", flush=True)


def mst_rollout_steps(epoch):
    # Curriculum of the fresh MST run
    for start, steps in ((10, 1), (20, 2), (30, 3), (40, 5), (50, 8)):
        if epoch < start:
            return steps
    return MAX_ROLLOUT_STEPS


def train_mst(model, training_data, model_save_directory, config, lj_params=None, score=None, shuffle=False):
    """The MST loop; score(epoch) is called after every 10th epoch. config is one barostat config or a list with one per sim."""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, last_epoch=-1)
    sim_configs = config if isinstance(config, list) else [config] * len(training_data)

    log = []
    model.train()
    optimizer.zero_grad()
    for epoch in range(MST_EPOCHS):
        t_start = time.perf_counter()
        rollout_steps = mst_rollout_steps(epoch)

        # Trackers
        total_acc_loss = 0
        train_samples = 0

        # Freeze normalizers
        if epoch == FREEZE_NORM_EPOCH:
            model.node_normalizer.frozen = True
            model.edge_normalizer.frozen = True
            model.output_normalizer.frozen = True

        order = list(zip(training_data, sim_configs))
        if shuffle:
            random.Random(cpb.SEED + epoch).shuffle(order)
        for sim, config in order:
            dump_period = config["default_skip"]
            dt = config["dt"]

            # Get equilibrium bond lengths
            r0 = sim[0].edge_attr[:, -2].to(DEVICE)
            model.r0 = r0

            for start_idx in range(TRAIN_LIMIT):
                optimizer.zero_grad()

                current_window_graphs = [sim[k].detach().to(DEVICE) for k in range(start_idx, start_idx + HISTORY + 1)]
                box_delta_x = current_window_graphs[-1].box_tensor[0] - current_window_graphs[-2].box_tensor[0]
                current_box_vel_y = estimate_initial_box_vel_y_accurate(*current_window_graphs[-3:], dump_period * dt)

                rollout_loss = 0
                for step in range(rollout_steps):
                    target_graph = sim[HISTORY + 1 + start_idx + step].to(DEVICE)
                    input_graph = build_velocity_graph_correction(current_window_graphs).to(DEVICE)
                    model_inputs = ModelInputs(current_window_graphs[-2], current_window_graphs[-1], target_graph)

                    model_output = model(input_graph, is_training=True)
                    pred_graph_next = model.update(model_inputs, model_output)
                    rollout_loss += huber_loss(model, model_output, model_inputs, is_training=True)

                    new_lx = pred_graph_next.box_tensor[0] + box_delta_x
                    new_ly, current_box_vel_y = update_box_y_thermodynamic(
                        positions=pred_graph_next.pos,
                        edge_index=model_inputs.cur_graph.edge_index,
                        edge_attr=model_inputs.cur_graph.edge_attr,
                        current_box=model_inputs.cur_graph.box_tensor,
                        r0=r0,
                        box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
                        W_y=config["C_coupling"] * pred_graph_next.num_nodes * ((dump_period * dt) ** 2),
                        damping=config["damping"] * pred_graph_next.num_nodes * (dump_period * dt),
                        stride_dt=dump_period * dt,
                        lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                        target_pressure=config["target_pressure"],
                        temperature=config["temperature"],
                    )

                    # Add new box and update edge_attr (and the LJ edges)
                    pred_graph_next.box_tensor = torch.stack([new_lx, new_ly])
                    edges = get_correct_edge_attr(pred_graph_next, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
                    if isinstance(edges, tuple):
                        pred_graph_next.edge_index, pred_graph_next.edge_attr = edges
                    else:
                        pred_graph_next.edge_attr = edges

                    # Update window: shift left, append new prediction
                    current_window_graphs.pop(0)
                    current_window_graphs.append(pred_graph_next.detach())

                final_loss = rollout_loss / rollout_steps
                final_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_acc_loss += final_loss.item()
                train_samples += 1

        model.save_checkpoint(os.path.join(model_save_directory, f"checkpoint_epoch_{epoch}.pt"))
        lr_scheduler.step()
        log.append({"epoch": epoch, "rollout_steps": rollout_steps, "train_loss": total_acc_loss / train_samples, "time": time.perf_counter() - t_start})
        pd.DataFrame(log).to_csv(os.path.join(model_save_directory, "train_log.csv"), index=False)
        print(f"Epoch {epoch:<3} | steps: {rollout_steps:<2} | loss: {log[-1]['train_loss']:.4e} | {log[-1]['time']:.2f} s.", flush=True)
        if score is not None and epoch % 10 == 9:
            score(epoch)
            model.train()


def evaluate_rollouts(model, test_data, envelope, checkpoints, barostat_config, lj_params=None, refine=cpb.no_refine):
    """Rolls out every test sim from its first HISTORY + 1 ground-truth frames and scores it at the checkpoints."""
    rows = []
    model.eval()
    for sim_idx, sim in enumerate(tqdm(test_data, desc="rollouts")):
        frames = [g.to(DEVICE) for g in sim[: HISTORY + 1]]
        r0 = sim[0].edge_attr[:, -2].to(DEVICE)
        model.r0 = r0

        # Compute dumping period (N MD steps in 1 dump step)
        sim_strain = (sim[1].box.x - sim[-1].box.x) / sim[0].box.x
        config = deepcopy(barostat_config)
        config["default_skip"] = int(int(sim_strain / 1e-5 / 0.01) / len(sim)) + 1
        box_delta_x = frames[-1].box_tensor[0] - frames[-2].box_tensor[0]
        box_vel_y = estimate_initial_box_vel_y_accurate(frames[-3], frames[-2], frames[-1], config["default_skip"] * config["dt"])

        cutoff = lj_params.cutoff if lj_params is not None else None
        rollout = cpb.constrained_rollout([model] * (HISTORY + 1), frames, checkpoints[-1], config, box_delta_x, box_vel_y, r0, refine, lj_params=lj_params)
        rows += [{"sim": sim_idx, **row} for row in cpb.evaluate(rollout, sim, HISTORY, checkpoints, envelope, cutoff)]
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    for step, d in df.groupby("step"):
        r2 = 1 - ((d.pred_p - d.gt_p) ** 2).sum() / ((d.gt_p - d.gt_p.mean()) ** 2).sum()
        print(f"step {step:>3} | nu R^2 {r2:.3f} | mean |nu error| {np.abs(d.pred_p - d.gt_p).mean():.3f} | pos MSE {d.mse.mean():.3e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="node_optimized")
    parser.add_argument("--model", choices=["plain", "residual", "harmonic"], required=True)
    parser.add_argument("--training", choices=["ost", "mst", "none"], default="ost")
    parser.add_argument("--checkpoint", help="checkpoint to load before training or scoring")
    parser.add_argument("--tag", help="output directory name, by default the model (and _mst)")
    parser.add_argument("--out-dir", default="residual_simulator")
    parser.add_argument("--n-train-sims", type=int, default=100)
    parser.add_argument("--n-test-sims", type=int, default=60)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    lj_params = dataset.get("lj_params")
    cutoff = lj_params.cutoff if lj_params is not None else None
    config = deepcopy(dataset["barostat"])
    config["default_skip"] = DUMP_PERIOD
    stride_dt = DUMP_PERIOD * config["dt"]

    torch.manual_seed(cpb.SEED)
    train_data, val_data, test_data = load_dataset(dataset, args.n_train_sims, args.n_test_sims)
    envelope = cpb.force_envelope(train_data, cutoff)  # before training, which moves frames between devices
    print(f"{args.dataset}: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test sims", flush=True)
    print(f"Harmonic step R^2 of the one-step acceleration (training sims): {harmonic_r2(train_data, stride_dt, cutoff):.4f}", flush=True)

    init_graph = Data(x=torch.ones((100, HISTORY * 2)), edge_attr=torch.ones((100, train_data[0][0].edge_attr.shape[1])))
    if args.model == "residual":
        model = ResidualModel(init_graph, 128, 2, 3, MASS, stride_dt, cutoff).to(DEVICE)
    elif args.model == "plain":
        model = VelocityModel(init_graph, 128, 2, 3).to(DEVICE)
    else:
        model = HarmonicOnly(stride_dt, cutoff)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    tag = args.tag or (args.model if args.training != "mst" else f"{args.model}_mst")
    model_save_directory = os.path.join(args.out_dir, tag)
    os.makedirs(model_save_directory, exist_ok=True)
    checkpoints = list(range(50, MAX_SIM_LEN - HISTORY, 50))

    def score(epoch):
        rows = evaluate_rollouts(model, test_data, envelope, checkpoints, config, lj_params)
        path = os.path.join(model_save_directory, "checkpoint_rollouts.csv")
        pd.DataFrame(rows).assign(epoch=epoch).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"Scored epoch {epoch}:", flush=True)
        summarize(rows)

    if args.training == "ost":
        train(model, train_data, val_data, model_save_directory)
    elif args.training == "mst":
        train_mst(model, train_data, model_save_directory, config, lj_params, score)

    rows = evaluate_rollouts(model, test_data, envelope, checkpoints, config, lj_params)
    pd.DataFrame(rows).assign(model=tag).to_csv(os.path.join(model_save_directory, "rollouts.csv"), index=False)
    summarize(rows)


if __name__ == "__main__":
    main()
