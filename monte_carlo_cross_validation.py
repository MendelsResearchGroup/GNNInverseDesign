import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
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


def get_explicit_splits(
    registry_path: str,
    target_data_type: str,
    train_config: dict,
    val_config: dict,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Explicitly draws N training samples and M validation samples based on predefined P-ranges.
    Everything left over defaults to the Test set.
    """
    df = pd.read_csv(registry_path)
    pool_df = df[df["data_type"] == str(target_data_type)].copy()

    if pool_df.empty:
        raise ValueError(f"No data found for data_type: {target_data_type}")

    # 1. Sample Training Data
    train_candidates = pool_df[
        (pool_df["poisson_ratio"] >= train_config["p_min"])
        & (pool_df["poisson_ratio"] < train_config["p_max"])
    ]

    if len(train_candidates) < train_config["count"]:
        raise ValueError(
            f"Train requested {train_config['count']}, but only {len(train_candidates)} available in range."
        )

    train_df = train_candidates.sample(n=train_config["count"], random_state=seed)

    # Remove the selected training data from the master pool to prevent leakage
    pool_df = pool_df.drop(train_df.index)

    # 2. Sample Validation Data
    val_candidates = pool_df[
        (pool_df["poisson_ratio"] >= val_config["p_min"])
        & (pool_df["poisson_ratio"] < val_config["p_max"])
    ]

    if len(val_candidates) < val_config["count"]:
        raise ValueError(
            f"Val requested {val_config['count']}, but only {len(val_candidates)} available in range."
        )

    val_df = val_candidates.sample(n=val_config["count"], random_state=seed)

    # Remove the selected validation data from the master pool
    pool_df = pool_df.drop(val_df.index)

    # 3. The Rest is Testing
    test_df = pool_df.copy()

    return train_df, val_df, test_df


def plot_try_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    try_num: int,
    save_dir: str,
):
    """Generates a 3-panel histogram visualizing the exact distributions used in the try."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    bins = np.linspace(-0.5, 0.5, 20)  # Adjust based on your actual physical P limits

    # Plot Train
    axes[0].hist(
        train_df["poisson_ratio"], bins=bins, color="skyblue", edgecolor="black"
    )
    axes[0].set_title(f"Train Set (n={len(train_df)})")
    axes[0].set_xlabel("Poisson's Ratio")
    axes[0].set_ylabel("Count")

    # Plot Validation
    axes[1].hist(
        val_df["poisson_ratio"], bins=bins, color="lightgreen", edgecolor="black"
    )
    axes[1].set_title(f"Validation Set (n={len(val_df)})")
    axes[1].set_xlabel("Poisson's Ratio")

    # Plot Test
    axes[2].hist(test_df["poisson_ratio"], bins=bins, color="salmon", edgecolor="black")
    axes[2].set_title(f"Test Set (n={len(test_df)})")
    axes[2].set_xlabel("Poisson's Ratio")

    fig.suptitle(f"Data Distribution for Try {try_num}", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"try_{try_num}_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved distribution histogram to {plot_path}")  # noqa: LOG015


def train_model_once_mst(
    data: dict[str, list],
    model: VelocityModel,
    epochs: int,
    barostat_config: dict,
    save_dir: str,
    device: str = "cuda",
) -> dict:

    freeze_norm_epoch = 5
    train_limit = 10
    accumulation_steps = 10
    learning_rate = 1e-3
    gamma = 0.995

    num_rollout_steps = 50
    history = 3

    # Limit training data
    logger.info(f"Using {len(data['train'])} simulations with for training.")
    logger.info(f"Using {len(data['val'])} simulations for validation.")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    results = {"training_loss": [], "validation_loss": [], "r2": [], "rollout_mse": []}

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
        for sim in data["train"]:
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
                box_delta_x = b1 - b0

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
                    raise Exception(f"Window size is too small : {len(current_window_graphs)}") # noqa: TRY002

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

                    new_lx = pred_graph_next.box_tensor[0] + box_delta_x
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

                total_acc_loss += (final_loss.item() * accumulation_steps)  # Re-multiply to get true magnitude for logging
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
                for val_sim in data["val"]:
                    input_graphs = [val_sim[i] for i in range(history + 1)]
                    
                    # Run basic rollout, no ITPO, no bootstrapping
                    rollout = get_rollout(
                        input_graphs=input_graphs,
                        gnn_simulator=model,
                        gnn_history=history,
                        num_steps=num_rollout_steps,
                        barostat_config=barostat_config,
                        device=device,
                    )

                    # Collect metrics
                    pred_ps.append(calc_p_ratio_box_tensor(rollout))
                    real_ps.append(calc_p_ratio_box_tensor(val_sim[: len(rollout)]))
                    position_mse = torch.nn.functional.mse_loss(rollout[-1].x, val_sim[len(rollout) - 1].x)
                    mses.append(position_mse.item())

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

        logger.info(
            f"Epoch {epoch + 1:>3} | "
            f"Train Loss: {avg_train_loss:.3e} | "
            f"Val Pos MSE: {avg_val_pos_mse:.3e} | "
            f"r2 : {r2:.3f} "
            f"Time: {t_stop - t_start:.2f} s"
        )

    return results


def train_model_once_ost(
    data: dict[str, list],
    gnn_simulator: VelocityModel,
    epochs: int,
    barostat_config: dict,
    save_dir: str,
    device: str = "cuda",
) -> dict:

    freeze_norm_epoch = 5
    train_limit = 15
    accumulation_steps = 10
    learning_rate = 1e-3
    gamma = 0.995

    num_rollout_steps = 50
    gnn_history = 3

    # Limit training data
    logger.info(f"Using {len(data['train'])} simulations with for training.")
    logger.info(f"Using {len(data['val'])} simulations for validation.")

    params = filter(lambda p: p.requires_grad, gnn_simulator.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    results = {"training_loss": [], "validation_loss": [], "r2": [], "rollout_mse": []}

    gnn_simulator.train()
    optimizer.zero_grad()

    r2 = 0.0
    for epoch in range(epochs):
        t_start = time.perf_counter()

        # Trackers
        total_acc_loss = 0
        total_val_loss = 0
        train_samples = 0

        # Freeze normalizers
        if epoch == freeze_norm_epoch:
            gnn_simulator.node_normalizer.frozen = True
            gnn_simulator.edge_normalizer.frozen = True
            gnn_simulator.output_normalizer.frozen = True

        gnn_simulator.train()
        for sim in data["train"]:

            starting_points = [i for i in range(train_limit)]
            for i, start_idx in enumerate(starting_points):
                indices = [step + start_idx for step in range(gnn_history + 1)]
                target_idx = gnn_history + 1 + start_idx

                current_window_graphs = [sim[k].detach().to(device) for k in indices]
                input_graph = build_velocity_graph_correction(current_window_graphs).to(device)

                model_inputs = ModelInputs(
                    current_window_graphs[-2].to(device),
                    current_window_graphs[-1].to(device),
                    sim[target_idx].to(device),
                )

                model_output = gnn_simulator(input_graph, is_training=True)

                acc_loss = huber_loss(gnn_simulator, model_output, model_inputs, is_training=True)
                loss_for_backward = acc_loss / accumulation_steps
                loss_for_backward.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(gnn_simulator.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_acc_loss += acc_loss.item()
                train_samples += 1

            # Catch remaining gradients for the sim if train_limit isn't divisible by accumulation_steps
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_acc_loss / train_samples
        results["training_loss"].append(avg_train_loss)

        # One-step validation for huber loss
        total_val_loss = 0
        val_samples = 0
        with torch.no_grad():
            gnn_simulator.eval()
            for val_sim in data["val"]:
                
                # Evaluate on the same trajectory window length as training
                starting_points = [i for i in range(train_limit)]
                for start_idx in starting_points:
                    indices = [step + start_idx for step in range(gnn_history + 1)]
                    target_idx = gnn_history + 1 + start_idx

                    current_window_graphs = [val_sim[k].detach().to(device) for k in indices]
                    input_graph = build_velocity_graph_correction(current_window_graphs).to(device)

                    model_inputs = ModelInputs(
                        current_window_graphs[-2],
                        current_window_graphs[-1],
                        val_sim[target_idx].to(device),
                    )

                    model_output = gnn_simulator(input_graph, is_training=False)
                    val_loss = huber_loss(gnn_simulator, model_output, model_inputs, is_training=False)

                    total_val_loss += val_loss.item()
                    val_samples += 1

        avg_val_loss = total_val_loss / val_samples
        results["validation_loss"].append(avg_val_loss)     

        # Rollout validation
        if epoch % 5 == 0:
            real_ps, pred_ps, mses = [], [], []
            with torch.no_grad():
                gnn_simulator.eval()
                for val_sim in data["val"]:
                    input_graphs = [val_sim[i] for i in range(gnn_history + 1)]
                    
                    # Run basic rollout, no ITPO, no bootstrapping
                    rollout = get_rollout(
                        input_graphs=input_graphs,
                        gnn_simulator=gnn_simulator,
                        gnn_history=gnn_history,
                        num_steps=num_rollout_steps,
                        barostat_config=barostat_config,
                        device=device,
                    )

                    # Collect metrics
                    pred_ps.append(calc_p_ratio_box_tensor(rollout))
                    real_ps.append(calc_p_ratio_box_tensor(val_sim[: len(rollout)]))
                    position_mse = torch.nn.functional.mse_loss(rollout[-1].x, val_sim[len(rollout) - 1].x)
                    mses.append(position_mse.item())

            r2 = r2_score(real_ps, pred_ps)
            results["r2"].append(r2)
            results["rollout_mse"].append(mses)

            # Calculate average MSE for logging
            if mses:
                avg_val_pos_mse = sum(mses) / len(mses)

            # Save model
            gnn_simulator.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        # End of epoch cleanup
        gnn_simulator.train()
        lr_scheduler.step()
        t_stop = time.perf_counter()

        # Output training and validation metrics
        logger.info(
            f"Epoch {epoch + 1:>3} | "
            f"Train Loss: {avg_train_loss:.3e} | "
            f"Val Loss: {avg_val_loss:.3e} | "
            f"Val Pos MSE: {avg_val_pos_mse:.3e} | "
            f"r2 : {r2:.3f} "
            f"Time: {t_stop - t_start:.2f} s"
        )

    return results


if __name__ == "__main__":
    # dePablo-style dataset
    dataset_type = DatasetType.NodeOptimized

    # Setup main directory
    main_directory = os.path.join("./cross_val", f"{dataset_type}", "MST")
    os.makedirs(main_directory, exist_ok=True)

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(main_directory, "training_crossval.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # Pick barostat params
    if dataset_type is DatasetType.NodeOptimized:
        barostat_config = barostat_parameters.node_optimizated
    elif dataset_type is DatasetType.StiffOptimized:
        barostat_config = barostat_parameters.stiff_optimized
    else:
        raise ValueError("Unreachable.")

    # Explicitly take 200 training sims with P > 0.1
    train_config = {"count": 200, "p_min": 0.1, "p_max": float("inf")}

    # Explicitly take 200 validation sims with P > 0.1
    val_config = {"count": 200, "p_min": 0.1, "p_max": float("inf")}

    # 1. Preload data into RAM once
    registry_path = "./data/data_registry.csv"
    df = pd.read_csv(registry_path)
    type_df = df[df["data_type"] == f"{dataset_type}"]

    all_files = type_df["file_path"].tolist()
    logger.info(f"Pre-loading {len(all_files)} total simulations into memory...")
    max_sim_len = 100

    # Dictionary mapping: { "path/to/file.pt" : tensor_data }
    master_tensors = {}
    for file in tqdm(all_files, desc="Loading & Preparing Data"):
        # Load and process
        sim = torch.load(file, weights_only=False)[:max_sim_len]
        prepared_sim = prepare_traj(sim, calc_angles=False)

        # Store in dict for instant retrieval later
        master_tensors[file] = prepared_sim

    logger.info("Data loaded successfully.")

    n_tries = 20
    try_results = []

    for try_num in range(n_tries):
        seed = 42 + try_num
        logger.info(f"\n{'=' * 40}\nStarting Try {try_num + 1}/{n_tries} (Seed: {seed})\n{'=' * 40}")

        # Get explicit data frames using the function from the previous step
        train_df, val_df, test_df = get_explicit_splits(
            registry_path=registry_path,
            target_data_type=dataset_type,
            train_config=train_config,
            val_config=val_config,
            seed=seed,
        )

        logger.info(f"Split Result | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        # Create a directory for this trye
        try_save_dir = os.path.join(main_directory, f"try_{try_num + 1}")
        os.makedirs(try_save_dir, exist_ok=True)

        plot_try_distributions(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            try_num=try_num + 1,
            save_dir=try_save_dir,
        )

        # Construct the data dictionary by looking up paths
        data = {
            "train": [master_tensors[path] for path in train_df["file_path"]],
            "val": [master_tensors[path] for path in val_df["file_path"]],
        }

        # Initialize a new simulator model
        epochs = 150
        mp_layers = 2
        mlp = 3
        hidden_size = 128
        history = 3
        device = "cuda"

        # Initialize a fresh model for this try
        init_graph = build_velocity_graph_correction(
            input_graphs=[data["train"][0][i].cpu().detach() for i in range(history + 1)],
            total_velocity=False,
            panic_at_positions=False,
        ).to(device)
        gnn_simulator = VelocityModel(init_graph, hidden_size, mp_layers, mlp)
        gnn_simulator.to(device)

        # Run training for this try
        result = train_model_once_mst(data, gnn_simulator, epochs, barostat_config, try_save_dir, device=device)

        # Save try result
        try_results.append(result)

        # Save trained model for this try
        gnn_simulator_savepath = os.path.join(
            try_save_dir,
            f"model_full_range_h{history}_nl{mp_layers}_mlp{mlp}_epochs{epochs}.pt",
        )
        gnn_simulator.save_checkpoint(gnn_simulator_savepath)

        # Save result for this trye
        try_result_savepath = os.path.join(try_save_dir, "result.pkl")
        torch.save(result, try_result_savepath)

        # Force immediate garbage collection to prevent GPU VRAM fragmentation
        del gnn_simulator
        torch.cuda.empty_cache()
        gc.collect()

    # Finally save combined results to the main directory
    try_results_savepath = os.path.join(main_directory, "monte_carlo_cv_results.pkl")
    torch.save(try_results, try_results_savepath)
