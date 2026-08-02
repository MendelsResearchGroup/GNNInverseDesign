import logging
import os
import sys
import time

import torch
from sklearn.metrics import r2_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

import barostat_parameters
from barostat_utils import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from graph_utils import LJInteractionParams, get_correct_edge_attr, prepare_traj
from itpo_weights import DatasetType
from pressure import compute_per_particle_forces
from simulator_SA_cpu_test import Model as VelocityModel
from training_utils import ModelInputs, huber_loss
from utils import (
    build_velocity_graph_correction,
    calc_p_ratio_box_tensor,
    get_rollout,
    load_and_split_dataset,
)


def train_model_once_mst(
    data: dict[str, list],
    model: VelocityModel,
    epochs: int,
    barostat_config: dict,
    lj_params: LJInteractionParams,
    save_dir: str,
    device: str,
) -> dict:

    freeze_norm_epoch = 5
    train_sims = 200
    val_sims = 100
    train_limit = 15
    accumulation_steps = 10
    learning_rate = 1e-3
    gamma = 0.995

    num_rollout_steps = 100
    history = 3

    # Limit training data
    training_data = data["train"][:train_sims]
    logger.info(f"Using {len(training_data)} simulations for training.")

    # Limit validation data
    validation_data = data["val"][:val_sims]
    logger.info(f"Using {len(validation_data)} simulations for validation.")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(params, lr=learning_rate, weight_decay=0.0)
    lr_scheduler = ExponentialLR(optimizer, gamma, last_epoch=-1)

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
        elif epoch >= 30 and epoch < 40:
            rollout_steps = 5
        elif epoch >= 40 and epoch < 50:
            rollout_steps = 7
        elif epoch >= 50 and epoch < 60:
            rollout_steps = 10
        else:
            rollout_steps = 10

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
                sim_strain = (sim[0].box.x - sim[-1].box.x) / sim[0].box.x
                assumed_rollout_length = int(sim_strain / 1e-5 / 0.01)
                dump_period = int(assumed_rollout_length / len(sim)) + 1

            barostat_config["default_skip"] = dump_period

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
                    raise Exception(f"Window size is too small : {len(current_window_graphs)}")  # noqa: TRY002

                rollout_loss = 0
                for step in range(rollout_steps):
                    target_idx = history + 1 + start_idx + step
                    target_graph = sim[target_idx].to(device)

                    input_graph = build_velocity_graph_correction(current_window_graphs).to(device)

                    model_inputs = ModelInputs(
                        current_window_graphs[-2],
                        current_window_graphs[-1],
                        target_graph,
                    )

                    model_output = model(input_graph, is_training=True)
                    pred_graph_next = model.update(model_inputs, model_output)

                    step_loss = huber_loss(model, model_output, model_inputs, is_training=True)
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
                        lj_cutoff=lj_params.cutoff,
                        target_pressure=barostat_config["target_pressure"],
                        temperature=barostat_config["temperature"],
                    )

                    new_box_tensor = torch.stack([new_lx, new_ly])
                    current_box_vel_y = new_vel_y

                    # Add new box
                    pred_graph_next.box_tensor = new_box_tensor

                    # Update edge_attr
                    function_output = get_correct_edge_attr(
                        pred_graph_next,
                        recompute_stiff=False,
                        lj_params=lj_params,
                        panic_at_nontensor_box=True,
                    )
                    if isinstance(function_output, torch.Tensor):
                        pred_graph_next.edge_attr = function_output

                    elif isinstance(function_output, tuple):
                        edge_index, edge_attr = function_output
                        pred_graph_next.edge_index = edge_index
                        pred_graph_next.edge_attr = edge_attr

                    pred_graph_next.forces = compute_per_particle_forces(
                        pred_graph_next, r0=r0.to(pred_graph_next.pos.device), cutoff=lj_params.cutoff
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

                # Re-multiply to get true magnitude for logging
                total_acc_loss += final_loss.item() * accumulation_steps
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
                for val_sim in validation_data:
                    input_graphs = [val_sim[i] for i in range(history + 1)]


                    rollout = get_rollout(
                        input_graphs=input_graphs,
                        gnn_simulator=model,
                        gnn_history=history,
                        num_steps=num_rollout_steps,
                        barostat_config=barostat_config,
                        lj_params=lj_params,
                        device=device,
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
        model.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
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
    # Special barostat params
    barostat_config = barostat_parameters.lj_noisy

    # Special dataset type
    dataset_type = DatasetType.LJNoisy

    # Typical split
    poisson_buckets = [
        {"min": 0.0, "max": 1.0, "count": 400},
]

    main_directory = os.path.join("./LJ_trained_models", f"{dataset_type}", "MST")
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

    # Load the files
    train_files, val_files, test_files = load_and_split_dataset(
        registry_path="./data/data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/data_registry.csv",
        target_data_type=dataset_type,
        possion_buckets=poisson_buckets,
        split_ratios=(0.5, 0.25, 0.25),
        seed=42,
    )

    # Load actual data
    data = {
        "train": {},
        "val": {},
        "test": {},
    }

    max_sim_len = 200
    logger.info(f"Loading data with maximum sim lengths {max_sim_len}.")
    for key in data:
        if key == "train":
            data[key] = [
                torch.load(file, weights_only=False)[:max_sim_len]
                for file in tqdm(train_files, desc=f"{key:<5} data")
            ]
        elif key == "val":
            data[key] = [
                torch.load(file, weights_only=False)[:max_sim_len]
                for file in tqdm(val_files, desc=f"{key:<5} data")
            ]
        elif key == "test":
            data[key] = [
                torch.load(file, weights_only=False)[:max_sim_len]
                for file in tqdm(test_files, desc=f"{key:<5} data")
            ]
        else:
            raise ValueError(f"Unexpected key in data dictionary: {key}. ")

    logger.info("Preparing data...")
    for data_type, sims in data.items():
        prepared = []
        for sim in tqdm(sims, desc=f"{data_type:<5} data"):
            lj_params = LJInteractionParams(0.01, 1.0, 1.122)
            prepared_sim = prepare_traj(sim, lj_params, calc_angles=False)
            prepared.append(prepared_sim)
        data[data_type] = prepared

    logger.info("Data loaded successfully: ")
    logger.info(f"Train data: {len(data['train'])} sims.")
    logger.info(f"Val data:   {len(data['val'])} sims.")
    logger.info(f"Test data:  {len(data['test'])} sims.")

    # Define model hyperparameters and number of training epochs
    epochs = 300
    mp_layers = 2
    mlp = 3
    hidden_size = 128
    history = 3
    device = "cuda"

    # Initialize a new simulator model
    init_graph = build_velocity_graph_correction(
        input_graphs=[data["train"][0][i].cpu().detach() for i in range(history + 1)],
        total_velocity=False,
        panic_at_positions=False,
    ).to(device)
    gnn_simulator = VelocityModel(init_graph, hidden_size, mp_layers, mlp)
    gnn_simulator.to(device)

    # Run training
    save_dir = os.path.join(main_directory, "trained_model")
    os.makedirs(save_dir, exist_ok=True)
    result = train_model_once_mst(
        data, gnn_simulator, epochs, barostat_config, lj_params, save_dir, device
    )

    # Save trained model
    gnn_simulator_savepath = os.path.join(
        save_dir,
        f"model_full_range_h{history}_nl{mp_layers}_mlp{mlp}_epochs{epochs}.pt",
    )
    gnn_simulator.save_checkpoint(gnn_simulator_savepath)

    # Save result for this fold
    fold_result_savepath = os.path.join(save_dir, "result.pkl")
    torch.save(result, fold_result_savepath)
