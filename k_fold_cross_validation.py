import gc
import logging
import os
import sys
import time
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import barostat_parameters
from barostat_utils import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from graph_utils import get_correct_edge_attr, prepare_traj
from itpo_weights import DatasetType
from pressure import compute_per_particle_forces
from simulator_SA_cpu_test import Model as VelocityModel
from training_utils import ModelInputs, huber_loss
from utils import build_velocity_graph_correction, calc_p_ratio_box_tensor, get_rollout


def get_sampled_files_with_labels(
    registry_path: str,
    target_data_type: DatasetType,
    poisson_buckets: list,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Samples data files and returns a tuple of:
    1. A shuffled list of file paths.
    2. A matching list of bucket integer IDs to use for stratification.
    """
    df = pd.read_csv(registry_path)
    type_df = df[df["data_type"] == str(target_data_type)]

    if type_df.empty:
        raise ValueError(f"No data found for data_type: {str(target_data_type)}")

    sampled_dfs = []

    for i, bucket in enumerate(poisson_buckets):
        p_min = bucket.get("min", float("-inf"))
        p_max = bucket.get("max", float("inf"))
        req_count = bucket["count"]

        bucket_df = type_df[
            (type_df["poisson_ratio"] >= p_min) & (type_df["poisson_ratio"] < p_max)
        ].copy()

        available = len(bucket_df)
        if available == 0:
            logger.warning(f"Bucket {i} ({p_min} <= P < {p_max}) is empty.")
            continue

        if req_count > available:
            req_count = available

        sampled = bucket_df.sample(n=req_count, random_state=seed)

        # Tag each row with its specific bucket index for stratification
        sampled["bucket_id"] = i
        sampled_dfs.append(sampled)

    if not sampled_dfs:
        raise ValueError("No data was sampled from any bucket. Check threshold logic.")

    # Combine and shuffle while maintaining the index match between files and bucket_ids
    combined_df = (
        pd.concat(sampled_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    )

    return combined_df["file_path"].tolist(), combined_df["bucket_id"].tolist()


def train_model_once_mst(
    data: dict[str, List],
    model: VelocityModel,
    epochs: int,
    barostat_config: Dict,
    save_dir: str,
    device: str = "cuda",
) -> Dict:

    freeze_norm_epoch = 5
    train_sims = 200
    val_sims = 200
    train_limit = 10
    accumulation_steps = 10
    learning_rate = 1e-3
    gamma = 0.995

    num_rollout_steps = 50
    history = 3

    # Limit training data
    poisson_threshold = 0.1
    training_data = [sim for sim in data["train"] if calc_p_ratio_box_tensor(sim) >= poisson_threshold][:train_sims]
    logger.info(f"Using {len(training_data)} simulations with Poisson's ratio >= {poisson_threshold} for training.")
    logger.info(f"Using {len(data['val'][:val_sims])} simulations for validation.")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    results = {"training_loss": [], "r2": [], "rollout_mse": []}

    model.train()
    optimizer.zero_grad()

    r2 = 0.0

    for epoch in range(epochs):
        t_start = time.perf_counter()

        if epoch < 10:
            rollout_steps = 1
        elif epoch >= 10 and epoch < 20:
            rollout_steps = 2
        elif epoch >= 20 and epoch < 30:
            rollout_steps = 3
        elif epoch >= 30:
            rollout_steps = 5
        else:
            rollout_steps = 5

        # Trackers
        total_acc_loss = 0
        train_samples = 0

        # Freeze normalizers
        if epoch == freeze_norm_epoch:
            model.node_normalizer.frozen = True
            model.edge_normalizer.frozen = True
            model.output_normalizer.frozen = True

        model.train()
        for sim in training_data:
            # Get equilibrium bond lengths
            r0 = sim[0].edge_attr[:, -2]

            starting_points = [i for i in range(train_limit)]
            if barostat_config["default_skip"] is not None:
                dump_period = barostat_config["default_skip"]
            else:
                sim_strain = (sim[1].box.x - sim[-1].box.x) / sim[0].box.x
                assumed_rollout_length = int(sim_strain / 1e-5 / 0.01)
                dump_period = int(assumed_rollout_length / len(sim)) + 1

            for i, start_idx in enumerate(starting_points):
                indices = [step + start_idx for step in range(history + 1)]
                current_window_graphs = [sim[k].detach().to(device) for k in indices]

                b0 = current_window_graphs[-2].box_tensor[0]
                b1 = current_window_graphs[-1].box_tensor[0]
                box_compression_factor = b1 / b0

                if len(current_window_graphs) < 3:
                    current_box_vel_y = estimate_initial_box_vel_y(
                        current_window_graphs[-2],
                        current_window_graphs[-1],
                        dump_period * barostat_config["dt"],
                    )
                elif len(current_window_graphs) >= 3:
                    current_box_vel_y = estimate_initial_box_vel_y_accurate(
                        current_window_graphs[-3],
                        current_window_graphs[-2],
                        current_window_graphs[-1],
                        dump_period * barostat_config["dt"],
                    )
                else:
                    raise Exception(f"Window size is too small : {len(current_window_graphs)}")

                rollout_loss = 0
                for step in range(rollout_steps):
                    target_idx = history + 1 + start_idx + step
                    target_graph = sim[target_idx].to(device)

                    input_graph = build_velocity_graph_correction(
                        current_window_graphs
                    ).to(device)

                    model_inputs = ModelInputs(
                        current_window_graphs[-2],
                        current_window_graphs[-1],
                        target_graph,
                    )

                    model_output = model(input_graph, is_training=True)
                    pred_graph_next = model.update(model_inputs, model_output)

                    step_loss = huber_loss(
                        model, model_output, model_inputs, is_training=True
                    )
                    rollout_loss += step_loss

                    dt = barostat_config["dt"]  # lammps dt
                    W_y = (
                        barostat_config["C_coupling"]
                        * pred_graph_next.num_nodes
                        * ((dump_period * dt) ** 2)
                    )
                    damping = (
                        barostat_config["damping"]
                        * pred_graph_next.num_nodes
                        * (dump_period * dt)
                    )

                    new_lx = pred_graph_next.box_tensor[0] * box_compression_factor
                    new_ly, new_vel_y = update_box_y_thermodynamic(
                        positions=pred_graph_next.pos,
                        edge_index=model_inputs.cur_graph.edge_index,
                        edge_attr=model_inputs.cur_graph.edge_attr,
                        current_box=model_inputs.cur_graph.box_tensor,
                        r0=r0.to(pred_graph_next.pos.device),
                        box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
                        W_y=W_y,
                        damping=damping,
                        stride_dt=dump_period * dt,
                        target_pressure=barostat_config["target_pressure"],
                        temperature=barostat_config["temperature"],
                    )

                    new_box_tensor = torch.stack([new_lx, new_ly])
                    current_box_vel_y = new_vel_y

                    # Add new box and update edge_attr
                    pred_graph_next.box_tensor = new_box_tensor
                    pred_graph_next.edge_attr = get_correct_edge_attr(
                        pred_graph_next,
                        recompute_stiff=False,
                        panic_at_nontensor_box=True,
                    )
                    pred_graph_next.forces = compute_per_particle_forces(
                        pred_graph_next, r0=r0.to(pred_graph_next.pos.device)
                    )

                    pred_graph_next_detached = pred_graph_next.detach()

                    # Update window: shift left, append new prediction
                    current_window_graphs.pop(0)
                    current_window_graphs.append(pred_graph_next_detached)

                # Divide by rollout_steps for average step loss, then by accumulation_steps
                final_loss = (rollout_loss / rollout_steps) / accumulation_steps
                final_loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_acc_loss += (
                    final_loss.item() * accumulation_steps
                )  # Re-multiply to get true magnitude for logging
                train_samples += 1

            # Catch remaining gradients for the sim if train_limit isn't divisible by accumulation_steps
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_acc_loss / train_samples
        results["training_loss"].append(avg_train_loss)

        avg_val_pos_mse = 0.0

        # Validation
        if epoch % 5 == 0:
            real_ps, pred_ps, mses = [], [], []
            with torch.no_grad():
                model.eval()
                for val_sim in data["val"][:val_sims]:
                    input_graphs = [val_sim[i] for i in range(history + 1)]
                    rollout = get_rollout(
                        input_graphs=input_graphs,
                        gnn_simulator=model,
                        gnn_history=history,
                        num_steps=num_rollout_steps,
                        barostat_config=barostat_config,
                        device="cuda",
                    )

                    pred_ps.append(calc_p_ratio_box_tensor(rollout))
                    real_ps.append(calc_p_ratio_box_tensor(val_sim[: len(rollout)]))
                    mses.append(
                        torch.nn.functional.mse_loss(
                            rollout[-1].x, val_sim[len(rollout) - 1].x
                        ).item()
                    )

            r2 = r2_score(real_ps, pred_ps)
            results["r2"].append(r2)
            results["rollout_mse"].append(mses)

            # Calculate average MSE for logging
            if mses:
                avg_val_pos_mse = sum(mses) / len(mses)

            # Save model
            model.save_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            )
            model.train()

        lr_scheduler.step()
        t_stop = time.perf_counter()

        # Logging replaced the print statement
        logger.info(
            f"Epoch {epoch + 1:>3} | "
            f"Train Loss: {avg_train_loss:.3e} | "
            f"Val Pos MSE: {avg_val_pos_mse:.3e} | "
            f"r2 : {r2:.3f} "
            f"Time: {t_stop - t_start:.2f} s"
        )

    return results


if __name__ == "__main__":
    # Constant barostat params
    barostat_config = barostat_parameters.node_optimizated

    # Constant split
    dataset_type = DatasetType.NodeOptimized
    poisson_buckets = [
        {"max": 0.1, "count": 500},  # P < 0.1
        {"min": 0.1, "max": 0.2, "count": 500},  # 0.1 <= P < 0.2
        {"min": 0.2, "count": 300},  # P >= 0.2
    ]

    main_directory = os.path.join("./cross_val", f"{dataset_type}", "MST")
    os.makedirs(main_directory, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(main_directory, "training_crossval.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # 1. Fetch paths along with their structural stratification labels
    all_files, bucket_labels = get_sampled_files_with_labels(
        registry_path="./data/data_registry.csv",
        target_data_type=dataset_type,
        poisson_buckets=poisson_buckets,
        seed=42,
    )

    # 2. Pre-load ALL data into memory exactly ONCE
    logging.info(f"Pre-loading {len(all_files)} simulations into memory...")
    max_sim_len = 100
    master_dataset = []

    for file in tqdm(all_files, desc="Loading & Preparing Data"):
        sim = torch.load(file, weights_only=False)[:max_sim_len]
        prepared_sim = prepare_traj(sim, calc_angles=False)
        master_dataset.append(prepared_sim)

    logger.info("Data loaded successfully.")

    k_folds = 10
    seed = 42
    # 3. Setup Stratified K-Fold Cross Validation
    # This ensures fold splits closely match the target bucket ratios
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_results = []

    # Pass bucket_labels into skf.split to enforce distribution balancing
    for fold, (train_idx, val_idx) in enumerate(skf.split(master_dataset, bucket_labels)):
        logger.info(f"\n{'=' * 40}\nStarting Fold {fold + 1}/{k_folds} (Stratified)\n{'=' * 40}")

        data = {
            "train": [master_dataset[i] for i in train_idx],
            "val": [master_dataset[i] for i in val_idx],
        }

        logging.info(f"Loaded {len(data['train'])} training and {len(data['val'])} validation simulations.")

        # Optional check: verify that the proportion of buckets in data['val'] remains uniform
        val_labels_in_fold = [bucket_labels[i] for i in val_idx]
        logger.info(
            f"Fold {fold + 1} - Validation sample counts per bucket: "
            f"{ {b_id: val_labels_in_fold.count(b_id) for b_id in set(bucket_labels)} }"
        )

        # Init a new simulator model
        epochs = 150
        mp_layers = 2
        mlp = 3
        hidden_size = 128
        history = 3
        device = "cuda"

        # Initialize a fresh model for this fold
        init_graph = build_velocity_graph_correction(
            input_graphs=[data["train"][0][i].cpu().detach() for i in range(history + 1)],
            total_velocity=False,
            panic_at_positions=False,
        ).to(device)
        gnn_simulator = VelocityModel(init_graph, hidden_size, mp_layers, mlp)
        gnn_simulator.to(device)

        # Create a directory for this fold
        fold_save_dir = os.path.join(main_directory, f"fold_{fold + 1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Run training for this fold
        result = train_model_once_mst(data, gnn_simulator, epochs, barostat_config, fold_save_dir, device=device)

        # Save fold result
        fold_results.append(result)

        # Save trained model for this fold
        gnn_simulator_savepath = os.path.join(fold_save_dir, f"model_full_range_h{history}_nl{mp_layers}_mlp{mlp}_epochs{epochs}.pt")
        gnn_simulator.save_checkpoint(gnn_simulator_savepath)

        # Save result for this fold
        fold_result_savepath = os.path.join(fold_save_dir, "result.pkl")
        torch.save(result, fold_result_savepath)

        # Force immediate garbage collection to prevent GPU VRAM fragmentation
        del gnn_simulator
        torch.cuda.empty_cache()
        gc.collect()

    # Finally save combined results to the main directory
    fold_results_savepath = os.path.join(main_directory, "k_fold_results.pkl")
    torch.save(fold_results, fold_results_savepath)
