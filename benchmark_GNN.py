import time
from copy import deepcopy
from enum import Enum, auto

import torch
import torch._dynamo
from torch_geometric.data import Data
from tqdm import tqdm

import barostat_parameters
from graph_utils import (
    LJInteractionParams,
    prepare_traj,
)
from itpo_weights import DatasetType
from pressure import compute_stress_curve, compute_virial_stress
from simulator_SA_cpu_test import Model as VelocityModel
from torch_simulator_wLJ_64 import DifferentiableCompression64
from training_utils import freeze_normalizer
from utils import (
    calc_p_ratio_box_tensor,
    get_rollout,
    load_and_split_dataset,
    simulate_then_rollout,
    to_f64,
)


class RolloutType(Enum):
    NoBootstrap = auto()
    Bootstrap = auto()

def run_gnn_simulator(
    num_steps: int,
    gnn_simulator: VelocityModel,
    gnn_history: int,
    test_sims: list[list[Data]],
    lj_params: LJInteractionParams | None,
    rollout_type: RolloutType,
    device: str = "cpu"
) -> dict[str, list]:

    if dataset_type is DatasetType.NodeOptimized:
        barostat_config = barostat_parameters.node_optimizated
    elif dataset_type is DatasetType.StiffOptimized:
        barostat_config = barostat_parameters.stiff_optimized
    elif dataset_type is DatasetType.StiffAngles:
        barostat_config = barostat_parameters.stiff_angles
    elif dataset_type is DatasetType.Noisy:
        barostat_config = barostat_parameters.noisy
    elif dataset_type is DatasetType.LJNoisy:
        barostat_config = barostat_parameters.lj_noisy

    gnn_simulator = gnn_simulator.eval()
    target_idx = num_steps + gnn_history + 1
    results = {
        "time_total_wall": [],
        "time_simulation_only": [],
        "mse": [],
        "pred_p": [],
        "gt_p": [],
        "gt_box": [],
        "pred_box": [],
        "gt_stress": [],
        "pred_stress": [],
    }

    t_start_wall = time.time()
    total_sim_time = 0.0

    for test_sim in tqdm(test_sims):

        # Compute dumping period (N MD steps in 1 dump step)        
        sim_strain = (test_sim[1].box.x - test_sim[-1].box.x) / test_sim[0].box.x
        assumed_rollout_length = int(sim_strain / 1e-5 / 0.01)
        dump_period = int(assumed_rollout_length / len(test_sim)) + 1

        barostat_config_current = deepcopy(barostat_config)
        barostat_config_current["default_skip"] = dump_period

        input_graphs = [g.cpu().detach() for g in test_sim[: gnn_history + 1]]

        # Generate a rollout
        t0_sim = time.time()

        match rollout_type:
            case RolloutType.NoBootstrap:
                rollout = get_rollout(
                    input_graphs=input_graphs,
                    gnn_simulator=gnn_simulator,
                    gnn_history=gnn_history,
                    num_steps=num_steps,
                    barostat_config=barostat_config_current,
                    lj_params=lj_params,
                    device=device
                )
            case RolloutType.Bootstrap:
                rollout = simulate_then_rollout(
                    starting_graph=input_graphs[0],
                    gnn_simulator=gnn_simulator,
                    gnn_history=gnn_history,
                    barostat_config=barostat_config_current,
                    lj_params=lj_params,
                    md_steps=(gnn_history * dump_period) + 1,
                    rollout_steps=num_steps,
                    device=device
                )
        total_sim_time += (time.time() - t0_sim)

        # Compare Predicted Position vs Ground Truth Position
        pos_mse = torch.stack([torch.nn.functional.mse_loss(rollout[i].pos.cpu(), test_sim[i].pos.cpu()) for i in range(len(rollout))])
        results["mse"].append(pos_mse.cpu().detach())
    
        pred_p = calc_p_ratio_box_tensor(rollout)
        results["pred_p"].append(pred_p.item())
        gt_p = calc_p_ratio_box_tensor(test_sim[:target_idx])
        results["gt_p"].append(gt_p.item())

        pred_box = torch.stack([g.box_tensor.cpu() for g in rollout])
        results["pred_box"].append(pred_box.cpu().detach())
        gt_box = torch.stack([g.box_tensor.cpu() for g in test_sim[:target_idx]])
        results["gt_box"].append(gt_box.cpu().detach())

        r0 = input_graphs[0].edge_attr[:, -2]
        gt_stress = torch.stack([compute_virial_stress(g, r0=r0.to(g.x.device)).cpu() for g in test_sim[:target_idx]], dim=0)
        results['gt_stress'].append(gt_stress.cpu().detach())
        pred_stress = torch.stack([compute_virial_stress(g, r0=r0.to(g.x.device)).cpu() for g in rollout], dim=0)
        results["pred_stress"].append(pred_stress.cpu().detach())

    results['time_total_wall'] = time.time() - t_start_wall
    results['time_simulation_only'] = total_sim_time

    return results

def run_torch_simulator(steps: int, input_sims: list[list[Data]], lj_params: LJInteractionParams | None, device: str = "cpu") -> dict[str, list]:
    results = {
        "time_total_wall": 0.0,
        "time_simulation_only": 0.0,
        "position_mse": [],
        "gt_box_change": [],
        "gt_box_p": [],
        "custom_box_p": [],
        "custom_box_change": [],
        "stress": [],
        "strain": [],
    }

    # Warmup phase (Eliminates torch.compile overhead)
    print("Warming up compiled simulator...")
    sim0 = input_sims[0]
    data_step0 = sim0[0].to(device)
    data_step0 = to_f64(data_step0)
    
    # Determine r0
    if data_step0.edge_attr.shape[1] == 4:
        r0 = data_step0.edge_attr[:, -2]
    elif data_step0.edge_attr.shape[1] == 7:
        r0 = data_step0.edge_attr[data_step0.edge_attr[:, 0] == 1][:, 4]
    
    dummy_simulator = DifferentiableCompression64(num_particles=sim0[0].num_nodes, lj_params=lj_params)

    # Run a short rollout to trigger the compiler
    _ = dummy_simulator.run_simulator(
        initial_data=data_step0, steps=2, r0=r0, debug=False, device=device
    )

    # Benchmark phase
    t_start_wall = time.time()
    total_sim_time = 0.0

    for sim in tqdm(input_sims):
        simulator = DifferentiableCompression64(num_particles=sim[0].num_nodes, lj_params=lj_params)

        strain = (sim[0].box.x - sim[-1].box.x) / sim[0].box.x
        assumed_rollout_length = int(strain / 1e-5 / 0.01)
        dump_period = int(assumed_rollout_length / len(sim)) + 1
        data_step0 = sim[0].to(device)
        data_step0 = to_f64(data_step0)
    
        if data_step0.edge_attr.shape[1] == 4:
            r0 = data_step0.edge_attr[:, -2]
        elif data_step0.edge_attr.shape[1] == 7:
            r0 = data_step0.edge_attr[data_step0.edge_attr[:, 0] == 1][:, 4]
        else:
             raise ValueError(f"Unexpected edge_attr dimension: {data_step0.edge_attr.shape[1]}")

        t0_sim = time.time()

        simulator_rollout, _conditions = simulator.run_simulator(
            initial_data=data_step0, steps=steps, r0=r0, debug=False, device=device,
        )

        total_sim_time += (time.time() - t0_sim)

        # Coarse grain to match GT Lammps 
        simulator_rollout = [step for i, step in enumerate(simulator_rollout) if i % dump_period == 0]
        target_idx = len(simulator_rollout) - 1

        # Collect metrics (Untimed by the sim timer, but included in wall time)
        results["gt_box_change"].append([graph.box_tensor.cpu().detach() for graph in sim[:target_idx]])
        results["custom_box_change"].append([graph.box_tensor.cpu().detach() for graph in simulator_rollout[:target_idx]])
        
        mse = torch.nn.functional.mse_loss(simulator_rollout[-1].x, sim[target_idx].x)
        results["position_mse"].append(mse.item())
        results["gt_box_p"].append(calc_p_ratio_box_tensor(sim, target_idx).item())
        results["custom_box_p"].append(calc_p_ratio_box_tensor(simulator_rollout, target_idx).item())
        
        stress_curve = compute_stress_curve(simulator_rollout, lj_cutoff=1.122, sample_stride=1)
        results["stress"].append(stress_curve.detach())
        
        final_strain = (sim[0].box_tensor[0] - sim[target_idx].box_tensor[0]) / sim[0].box_tensor[0]
        results["strain"].append(final_strain.item())
    
    results['time_total_wall'] = time.time() - t_start_wall
    results['time_simulation_only'] = total_sim_time

    return results


poisson_buckets = [
    {"min": -1.0, "max": 1.0, "count": 100}, # Full range
]

max_sim_len = 100

#region Default data

dataset_type = DatasetType.NodeOptimized

train_files, val_files, test_files = load_and_split_dataset(
    registry_path="./data/new_node_optimized/data_registry.csv",
    target_data_type=dataset_type,
    possion_buckets=poisson_buckets,
    split_ratios=(0.9, 0.05, 0.05),
    seed=42
)

# Load actual data
default_data = {
    'train': {},
    'val' : {},
    'test' : {},
}

print("Loading default data...")
for key in default_data:
    if key == 'train':
        default_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(train_files, desc=f"{key:<5} data")]
    elif key == 'val':
        default_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(val_files, desc=f"{key:<5} data")]
    elif key == 'test':
        default_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(test_files, desc=f"{key:<5} data")]
    else:
        raise ValueError(f"Unexpected key in data dictionary: {key}. ")

print("\nPreparing data...")
for data_type, sims in default_data.items():
    prepared = []
    for sim in tqdm(sims, desc=f"{data_type:<5} data"):
        prepared_sim = prepare_traj(sim, lj_params=None, calc_angles=False)
        prepared.append(prepared_sim)
    default_data[data_type] = prepared

#endregion

#region LJ data

dataset_type = DatasetType.LJNoisy

train_files, val_files, test_files = load_and_split_dataset(
    registry_path="./data/data_LJ_noisy_eps0.01_sigma1.0_cutoff1.122/data_registry.csv",
    target_data_type=dataset_type,
    possion_buckets=poisson_buckets,
    split_ratios=(0.9, 0.05, 0.05),
    seed=42
)

# Load actual data
lj_data = {
    'train': {},
    'val' : {},
    'test' : {},
}

print("\nLoading LJ data...")
for key in lj_data:
    if key == 'train':
        lj_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(train_files, desc=f"{key:<5} data")]
    elif key == 'val':
        lj_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(val_files, desc=f"{key:<5} data")]
    elif key == 'test':
        lj_data[key] = [torch.load(file, weights_only=False)[:max_sim_len] for file in tqdm(test_files, desc=f"{key:<5} data")]
    else:
        raise ValueError(f"Unexpected key in data dictionary: {key}. ")

print("\nPreparing data...")
default_params = LJInteractionParams(epsilon=0.01, sigma=1.0, cutoff=1.122)
for data_type, sims in lj_data.items():
    prepared = []
    for sim in tqdm(sims, desc=f"{data_type:<5} data"):
        prepared_sim = prepare_traj(sim, default_params, calc_angles=False)
        prepared.append(prepared_sim)
    lj_data[data_type] = prepared

#endregion

default_sizes = [sim[0].num_nodes for sim in default_data['train'] if sim[0].num_nodes <= 150][:40]
lj_sizes = [sim[0].num_nodes for sim in lj_data['train'] if sim[0].num_nodes <= 150][:40]

default_sims = [sim for sim in default_data['train'] if sim[0].num_nodes <= 150][:40]
lj_sims = [sim for sim in lj_data['train'] if sim[0].num_nodes <= 150][:40]

print(f"Default data: {len(default_sims)} sims")
print(f"LJ Data:      {len(lj_sims)} sims")

# Load models
device = "cpu"
gnn_history = 3
mp_layers = 2
mlp = 3
hidden_size = 128

models = {}

init_graph = Data(x=torch.ones((100, gnn_history*2)), edge_attr=torch.ones((100, 4)))
model: VelocityModel = VelocityModel(init_graph, hidden_size, mp_layers, mlp).to(device)
model_path = "./new_trained_models/node_optimized/MST/checkpoint_epoch_120.pt"
model.load_checkpoint(model_path)
model = freeze_normalizer(model)
models[DatasetType.NodeOptimized] = model

init_graph = Data(x=torch.ones((100, gnn_history*2)), edge_attr=torch.ones((100, 7)))
model: VelocityModel = VelocityModel(init_graph, hidden_size, mp_layers, mlp).to(device)
model_path = "./LJ_trained_models/lj_noisy/MST/trained_model/checkpoint_epoch_87.pt"
model.load_checkpoint(model_path)
model = freeze_normalizer(model)
models[DatasetType.LJNoisy] = model

steps = 10

# Flush the cache
print("\nFlushing torch.compile cache...")
torch._dynamo.reset()

# Benchmark default dataset
print("Benchmarking Default data, No Bootstrap")
default_noboot_results = run_gnn_simulator(
    steps, models[DatasetType.NodeOptimized], gnn_history, default_sims, None, RolloutType.NoBootstrap, device
)

# Flush the cache
print("\nFlushing torch.compile cache...")
torch._dynamo.reset()

# Benchmark default dataset
print("Benchmarking Default data, No Bootstrap")
default_boot_results = run_gnn_simulator(
    steps, models[DatasetType.NodeOptimized], gnn_history, default_sims, None, RolloutType.Bootstrap, device
)

# Flush the cache
print("\nFlushing torch.compile cache...")
torch._dynamo.reset()

# Benchmark the LJ dataset
print("Benchmarking LJ Data, No Bootstrap")
lj_noboot_results = run_gnn_simulator(steps, models[DatasetType.LJNoisy], gnn_history, lj_sims, default_params, RolloutType.NoBootstrap, device)

# Flush the cache
print("\nFlushing torch.compile cache...")
torch._dynamo.reset()

# Benchmark the LJ dataset
print("Benchmarking LJ Data, No Bootstrap")
lj_boot_results = run_gnn_simulator(steps, models[DatasetType.LJNoisy], gnn_history, lj_sims, default_params, RolloutType.Bootstrap, device)

print(f"Total wall time: {default_noboot_results['time_total_wall']:.2f} s.")
print(f"Total wall time: {default_boot_results['time_total_wall']:.2f} s.")
print(f"Total wall time: {lj_noboot_results['time_total_wall']:.2f} s.")
print(f"Total wall time: {lj_boot_results['time_total_wall']:.2f} s.")

print(f"Sim only time: {default_noboot_results['time_simulation_only']:.2f} s.")
print(f"Sim only time: {default_boot_results['time_simulation_only']:.2f} s.")
print(f"Sim only time: {lj_noboot_results['time_simulation_only']:.2f} s.")
print(f"Sim only time: {lj_boot_results['time_simulation_only']:.2f} s.")
