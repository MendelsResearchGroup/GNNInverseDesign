from itertools import batched

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from torch import Tensor
from torch_geometric.data import Data

from barostat_utils import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from graph_utils import LJInteractionParams, get_correct_edge_attr
from itpo_weights import DatasetType, ITPOWeights
from pressure import (
    compute_per_particle_forces,
    compute_potential_energy,
    compute_virial_stress,
)

# from torch_simulator_64 import DifferentiableCompression64
from torch_simulator_wLJ_64 import DifferentiableCompression64
from training_utils import GNNModel, ModelInputs


# Functions dealing with datasets
def load_and_split_dataset(
    registry_path: str,
    target_data_type: DatasetType,
    possion_buckets: list,
    split_ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42
):
    if sum(split_ratios) != 1.0:
        raise ValueError(f"expected `split_ratios` to sum up to 1.0, got {sum(split_ratios)}. ")
        
    df = pd.read_csv(registry_path)
    
    # Filter by data_type
    type_df = df[df['data_type'] == str(target_data_type)]
    
    if type_df.empty:
        raise ValueError(f"No data found for data_type: {target_data_type}")

    sampled_dfs = []

    # Process each requested bucket
    for i, bucket in enumerate(possion_buckets):
        p_min = bucket.get('min', float('-inf'))
        p_max = bucket.get('max', float('inf'))
        req_count = bucket['count']

        # Filter for the specific Poisson's ratio range (inclusive min, exclusive max)
        bucket_df = type_df[(type_df['poisson_ratio'] >= p_min) & (type_df['poisson_ratio'] < p_max)]
        
        available = len(bucket_df)
        if available == 0:
            print(f"Warning: Bucket {i} ({p_min} <= P < {p_max}) is empty.")
            continue
            
        if req_count > available:
            print(f"Warning: Bucket {i} requested {req_count} but only has {available}. Taking all {available}.")
            req_count = available
            
        # Sample the requested amount
        sampled = bucket_df.sample(n=req_count, random_state=seed)
        sampled_dfs.append(sampled)

    if not sampled_dfs:
        raise ValueError("No data was sampled from any bucket. Check threshold logic.")

    # Combine all buckets into one grand dataset and shuffle it
    combined_df = pd.concat(sampled_dfs)
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate split indices
    total_samples = len(combined_df)
    train_end = int(total_samples * split_ratios[0])
    val_end = train_end + int(total_samples * split_ratios[1])

    # Slice the dataframe into train, val, test
    train_df = combined_df.iloc[:train_end]
    val_df = combined_df.iloc[train_end:val_end]
    test_df = combined_df.iloc[val_end:]

    # Extract file paths
    def build_paths(subset_df):
        return [fname for fname in subset_df['file_path']]

    return build_paths(train_df), build_paths(val_df), build_paths(test_df)


def split_sims(data: list[list[Data]], segment_length: int, step_limit: int) -> list[list[Data]]:
    """Batch simulations into tuples of n graphs.
    Preserves some time relations but makes it more random for training."""

    new_data = []
    for sim in data:
        new_data += batched(sim[:step_limit], segment_length)
    return new_data


def visualize_nu_disribution(data: dict[str, list[list[Data]]]):
    ps = {}
    all_values = []

    for data_type, sims in data.items():
        ps[data_type] = [calc_p_ratio_box_tensor(sim).item() for sim in sims]
        all_values.extend(ps[data_type])

    min_val = min(all_values)
    max_val = max(all_values)
    common_bins = np.linspace(min_val, max_val, 30) 

    for data_type, values in ps.items():
        plt.hist(
        values, 
        bins=common_bins, 
        edgecolor='black', 
        alpha=0.6, 
        label=f"{data_type} data"
    )

    plt.title("$\\nu$ distribution")
    plt.xlabel("GT LAMMPS $\\nu$")
    plt.ylabel("N")
    plt.legend()
    plt.show()


# Functions dealing with position graphs
def radius_graph(graph: Data, r: float) -> Tensor:
    G = nx.Graph()
    nodes = [(idx, {"pos": node}) for idx, node in enumerate(graph.x)]
    G.add_nodes_from(nodes)
    edge_index = torch.tensor(nx.geometric_edges(G, r)).T
    return edge_index


# Functions for graph visualization
def draw_graph(
    graph: Data | list[Data],
    edges: bool = True,
    show_lj: bool = True,
    periodic_edges: bool = True,
    box: bool = False,
    node_color: str = "skyblue",
    node_size: float = 20,
    node_labels: bool = False,
    chosen_nodes: list | None = None,
    chosen_nodes_color: str = "red",
):
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    graph = graph.cpu().detach()
    G = nx.Graph()

    # Extract box dimensions for Minimum Image Convention calculations
    if hasattr(graph, 'box_tensor') and graph.box_tensor is not None:
        Lx, Ly = graph.box_tensor[0].item(), graph.box_tensor[1].item()
    elif hasattr(graph, 'box') and graph.box is not None:
        Lx = graph.box.x2 - graph.box.x1
        Ly = graph.box.y2 - graph.box.y1
    else:
        # Fallback if no explicit box is provided
        Lx = (torch.max(graph.x[:, 0]) - torch.min(graph.x[:, 0])).item()
        Ly = (torch.max(graph.x[:, 1]) - torch.min(graph.x[:, 1])).item()

    # Add nodes and extract positions
    pos = {}
    for index, node in enumerate(graph.x):
        G.add_node(index)
        pos[index] = (node[0].item(), node[1].item())

    harmonic_edgelist = []
    lj_edgelist = []
    
    # Store periodic edges as (u, v, dx_mic, dy_mic)
    harmonic_pbc_list = []
    lj_pbc_list = []

    if edges and hasattr(graph, 'edge_index') and graph.edge_index is not None:
        edge_index = graph.edge_index.numpy().T
        has_edge_attr = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
        num_features = graph.edge_attr.shape[1] if has_edge_attr else 0
        
        for i, edge in enumerate(edge_index):
            u, v = edge[0], edge[1]
            
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]

            # Detect boundary crossing
            is_pbc_cross = abs(dx) > Lx / 2 or abs(dy) > Ly / 2

            # Determine if it's an LJ edge based on shape & one-hot vector
            is_lj = False
            if num_features == 7 and graph.edge_attr[i, 1] == 1.0:
                is_lj = True

            if is_pbc_cross:
                if not periodic_edges:
                    continue
                else:
                    # Calculate Minimum Image Convention (MIC) vector
                    dx_mic = dx - Lx * round(dx / Lx)
                    dy_mic = dy - Ly * round(dy / Ly)
                    
                    if is_lj:
                        if show_lj:
                            lj_pbc_list.append((u, v, dx_mic, dy_mic))
                    else:
                        harmonic_pbc_list.append((u, v, dx_mic, dy_mic))
            else:
                if is_lj:
                    if show_lj:
                        lj_edgelist.append((u, v))
                else:
                    harmonic_edgelist.append((u, v))

    if box and hasattr(graph, 'box'):
        B = nx.Graph()
        box_corners = [i for i in range(len(graph.x) + 1, len(graph.x) + 5)]
        box_edges = [
            [box_corners[0], box_corners[1]],
            [box_corners[1], box_corners[2]],
            [box_corners[2], box_corners[3]],
            [box_corners[3], box_corners[0]],
        ]
        b_pos = {
            box_corners[0]: (graph.box.x1, graph.box.y2),
            box_corners[1]: (graph.box.x2, graph.box.y2),
            box_corners[2]: (graph.box.x2, graph.box.y1),
            box_corners[3]: (graph.box.x1, graph.box.y1)
        }
        for corner, edge in zip(box_corners, box_edges):
            B.add_node(corner)
            B.add_edge(edge[0], edge[1])
        nx.draw_networkx(B, b_pos, with_labels=False, node_color="black", node_size=1)

    # Nodelist and colors
    nodelist = list(G)
    node_colors = [node_color for _ in range(len(nodelist))]
    if chosen_nodes:
        for node in chosen_nodes:
            node_colors[node] = chosen_nodes_color

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, node_color=node_colors, node_size=node_size
    )

    # Draw Standard Edges
    if edges:
        if harmonic_edgelist:
            nx.draw_networkx_edges(G, pos, edgelist=harmonic_edgelist, edge_color="black", width=1.0)
        
        if show_lj and lj_edgelist:
            nx.draw_networkx_edges(G, pos, edgelist=lj_edgelist, edge_color="tomato", width=1.2, style="dashed", alpha=0.4)

    # Helper function to draw periodic half-bonds
    def draw_pbc_collection(pbc_list, color, width, style, alpha=1.0):
        segments = []
        for u, v, dx_mic, dy_mic in pbc_list:
            x_u, y_u = pos[u]
            x_v, y_v = pos[v]
            # Draw from u extending exactly halfway along the MIC vector
            segments.append([(x_u, y_u), (x_u + 0.5 * dx_mic, y_u + 0.5 * dy_mic)])
            # Draw from v extending backwards halfway along the MIC vector
            segments.append([(x_v, y_v), (x_v - 0.5 * dx_mic, y_v - 0.5 * dy_mic)])
            
        if segments:
            lc = LineCollection(segments, colors=color, linewidths=width, linestyles=style, alpha=alpha)
            ax.add_collection(lc)

    # Draw Periodic Edges
    if periodic_edges:
        draw_pbc_collection(harmonic_pbc_list, color="black", width=1.0, style="solid")
        if show_lj:
            draw_pbc_collection(lj_pbc_list, color="tomato", width=1.2, style="dashed", alpha=0.4)

    if node_labels:
        labels = {i: str(i) for i in range(len(graph.x))}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.axis("equal")
    plt.show()


def draw_two_graphs(
    graph1: Data,
    graph2: Data,
    edges: bool = True,
    periodic_edges: bool = False,
    box: bool = False,
    node_color: str = "skyblue",
    node_size: float = 20,
    node_labels: bool = False,
    chosen_nodes1: list | None = None,
    chosen_nodes2: list | None = None,
    chosen_nodes_color: str = "red",
    titles: list[str] | None = None,
):
    """
    Plots two PyTorch Geometric graphs side-by-side using the specified visualization logic.
    """

    # Create a subplot with 1 row and 2 columns
    _fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Prepare lists to iterate over
    graphs = [graph1, graph2]
    chosen_nodes_list = [chosen_nodes1, chosen_nodes2]

    # Inner function to handle the plotting logic for a single axis
    def _plot_on_axis(ax, graph, chosen_nodes, title=None):
        graph = graph.cpu().detach()
        G = nx.Graph()

        # Add nodes
        for index, node in enumerate(graph.x):
            G.add_node(index)

        # Add edges
        if edges:
            edge_index = graph.edge_index.numpy().T
            for edge in edge_index:
                # Calculate edge length
                edge_len = torch.norm(graph.x[edge[1]] - graph.x[edge[0]])

                # Periodic edge logic
                if not periodic_edges:
                    # Check X and Y dimensions for periodicity
                    x_span = torch.max(graph.x[:, 0]) - torch.min(graph.x[:, 0])
                    y_span = torch.max(graph.x[:, 1]) - torch.min(graph.x[:, 1])

                    if (edge_len > x_span / 2) or (edge_len > y_span / 2):
                        continue
                    else:
                        G.add_edge(edge[0], edge[1])
                else:
                    G.add_edge(edge[0], edge[1])

        pos = {index: (node[0].item(), node[1].item()) for index, node in enumerate(graph.x)}

        # Draw Box
        if box:
            B = nx.Graph()
            # Create unique IDs for box corners to avoid clashing with graph nodes
            start_id = len(graph.x) + 1
            box_corners = [i for i in range(start_id, start_id + 4)]
            box_edges = [
                [box_corners[0], box_corners[1]],
                [box_corners[1], box_corners[2]],
                [box_corners[2], box_corners[3]],
                [box_corners[3], box_corners[0]],
            ]
            for corner, edge in zip(box_corners, box_edges):
                B.add_node(corner)
                B.add_edge(edge[0], edge[1])

            b_pos = {}
            b_pos[box_corners[0]] = (graph.box.x1, graph.box.y2)
            b_pos[box_corners[1]] = (graph.box.x2, graph.box.y2)
            b_pos[box_corners[2]] = (graph.box.x2, graph.box.y1)
            b_pos[box_corners[3]] = (graph.box.x1, graph.box.y1)

            # Pass the specific axis 'ax' to draw_networkx
            nx.draw_networkx(B, b_pos, with_labels=False, node_color="black", node_size=1, ax=ax)

        # Labels
        labels = {i: str(i) for i in range(len(graph.x))} if node_labels else None

        # Node coloring
        nodelist = list(G)
        current_node_colors = [node_color for _ in range(len(nodelist))]

        if chosen_nodes:
            for node_idx in chosen_nodes:
                # Ensure node_idx is within bounds
                if node_idx < len(current_node_colors):
                    current_node_colors[node_idx] = chosen_nodes_color

        # Draw main graph on the specific axis
        nx.draw_networkx(G, pos, with_labels=node_labels, labels=labels, node_color=current_node_colors, node_size=node_size, font_size=10, ax=ax)

        if title:
            ax.set_title(title)

        # Keep aspect ratio equal so the physics looks correct
        ax.set_aspect("equal")
        ax.axis("off")

    # Loop through the two graphs and axes
    for i, ax in enumerate(axes):
        title = titles[i] if titles and len(titles) > i else None
        _plot_on_axis(ax, graphs[i], chosen_nodes_list[i], title)

    plt.tight_layout()
    plt.show()


def visualize_graphs(
    original_graph: Data | list[Data],
    predicted_graph: Data | list[Data],
    edges: bool = True,
    periodic_edges: bool = False,
    box: bool = False,
    node_size: float = 20,
    node_labels: bool = False,
    chosen_nodes: list | None = None,
    chosen_nodes_color: str = "green"
):
    # Handle list inputs safely
    if isinstance(original_graph, list):
        original_graph = original_graph[0]
    if isinstance(predicted_graph, list):
        predicted_graph = predicted_graph[0]

    original_graph = original_graph.cpu().detach()
    predicted_graph = predicted_graph.cpu().detach()

    plt.figure(figsize=(10, 10))

    def plot_single_graph(graph: Data, base_node_color: str, edge_color: str, alpha: float):
        lx, ly = graph.box_tensor[0].item(), graph.box_tensor[1].item()
        
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))

        if edges:
            # 1. Filter out LJ edges (keep only harmonic)
            is_harmonic = graph.edge_attr[:, 0].bool()
            edge_index = graph.edge_index[:, is_harmonic]

            # 2. Vectorized Periodic Edge Filtering
            if not periodic_edges:
                src, dst = edge_index
                dx = torch.abs(graph.x[src, 0] - graph.x[dst, 0])
                dy = torch.abs(graph.x[src, 1] - graph.x[dst, 1])
                
                # Edges are considered non-periodic if they don't cross half the box size
                mask = (dx <= lx / 2) & (dy <= ly / 2)
                edge_index = edge_index[:, mask]

            # 3. Instant NetworkX injection
            G.add_edges_from(edge_index.T.tolist())

        # Map node positions
        pos = {i: (graph.x[i, 0].item(), graph.x[i, 1].item()) for i in range(graph.num_nodes)}
        
        # Color specific chosen nodes
        node_colors = [base_node_color] * graph.num_nodes
        if chosen_nodes:
            for n in chosen_nodes:
                node_colors[n] = chosen_nodes_color

        labels = {i: str(i) for i in range(graph.num_nodes)} if node_labels else None

        nx.draw_networkx(
            G,
            pos,
            with_labels=node_labels,
            labels=labels,
            node_color=node_colors,
            edge_color=edge_color,
            node_size=node_size,
            font_size=10,
            alpha=alpha
        )
        return lx, ly

    # 1. Plot Original Graph (Blue nodes, Lightblue edges, solid opacity)
    lx, ly = plot_single_graph(
        original_graph, 
        base_node_color="blue", 
        edge_color="lightblue", 
        alpha=1.0
    )

    # 2. Plot Predicted Graph (Red nodes, Lightcoral edges, slightly transparent)
    plot_single_graph(
        predicted_graph, 
        base_node_color="red", 
        edge_color="lightcoral", 
        alpha=0.6
    )

    # 3. Draw Simulation Box
    if box:
        # Check if coordinates are centered around 0 (e.g. -lx/2 to lx/2) or bottom-left anchored (0 to lx)
        min_x = original_graph.x[:, 0].min().item()
        min_y = original_graph.x[:, 1].min().item()
        
        offset_x = -lx / 2 if min_x < -lx / 4 else 0.0
        offset_y = -ly / 2 if min_y < -ly / 4 else 0.0

        box_x = [offset_x, offset_x + lx, offset_x + lx, offset_x, offset_x]
        box_y = [offset_y, offset_y, offset_y + ly, offset_y + ly, offset_y]

        plt.plot(box_x, box_y, color="black", linestyle="--", linewidth=1.5, label="Simulation Box")
        plt.legend()

    plt.show()


# Functions for simulator model
def build_mlp(in_size: int, hidden_size: int, out_size: int, num_mlp: int = 3, lay_norm: bool = False, gaussian: bool = True) -> torch.nn.Sequential:
    layers = []
    # First linear layer
    layer = torch.nn.Linear(in_size, hidden_size)
    if gaussian:
        # Initialize weights and biases with Gaussian distribution
        torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / (in_size**0.5))  # std = 1/sqrt(n)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0 / (in_size**0.5))  # Biases are often initialized similarly
    layers.append(layer)
    layers.append(torch.nn.ReLU())

    # Add num_mlp-1 more pairs of linear layer and ReLU
    for _ in range(num_mlp - 2):
        layer = torch.nn.Linear(hidden_size, hidden_size)
        # Initialize weights and biases with Gaussian distribution
        if gaussian:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / (hidden_size**0.5))  # std = 1/sqrt(n)
            torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0 / (hidden_size**0.5))  # Biases
        layers.append(layer)
        layers.append(torch.nn.ReLU())

    # Final output layer
    layer = torch.nn.Linear(hidden_size, out_size)
    if gaussian:
        # Initialize weights and biases with Gaussian distribution
        torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / (hidden_size**0.5))  # std = 1/sqrt(n)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0 / (hidden_size**0.5))  # Biases
    layers.append(layer)

    if lay_norm:
        layers.append(torch.nn.LayerNorm(normalized_shape=out_size))

    # Create the model
    module = torch.nn.Sequential(*layers)
    return module


def get_affine_residual_velocity(curr_graph: Data, target_graph: Data) -> tuple[Tensor, Tensor]:
    strain: Tensor = (curr_graph.box_tensor[0] - target_graph.box_tensor[0]) / curr_graph.box_tensor[0]

    # affine velocity due to compression
    affine_pos_x = curr_graph.x[:, 0] * (1 - strain)
    affine_velocity_x = affine_pos_x - curr_graph.x[:, 0]
    affine_velocity_y = torch.zeros_like(affine_velocity_x)
    affine_velocity = torch.column_stack((affine_velocity_x, affine_velocity_y))

    # residual velocity
    global_velocity = target_graph.x - curr_graph.x
    residual_velocity = global_velocity - affine_velocity

    return affine_velocity, residual_velocity


def build_velocity_graph_correction(input_graphs: list[Data], total_velocity: bool = False, panic_at_positions: bool = False) -> Data:
    base_graph = input_graphs[-1]

    if len(input_graphs) == 1 and not panic_at_positions:
        return Data(
            x=base_graph.x,
            pos=base_graph.x,
            affine_velocities=torch.zeros_like(base_graph.x),
            edge_index=base_graph.edge_index,
            edge_attr=base_graph.edge_attr,
            box=base_graph.box if hasattr(base_graph, "box") else None,
            box_tensor=base_graph.box_tensor if hasattr(base_graph, "box_tensor") else None,
            t=base_graph.t if hasattr(base_graph, "t") else None,
        )
    elif len(input_graphs) == 1 and panic_at_positions:
        raise ValueError("Only one graph in input_graphs, can't construct velocity input.")

    affine_velocities = []
    residual_velocities = []
    for i in range(len(input_graphs) - 1, 0, -1):
        target_graph = input_graphs[i]
        curr_graph = input_graphs[i - 1]
        affine_velocity, residual_velocity = get_affine_residual_velocity(curr_graph, target_graph)
        affine_velocities.append(affine_velocity)
        residual_velocities.append(residual_velocity)

    residual_velocities = torch.column_stack(residual_velocities)
    affine_velocities = torch.column_stack(affine_velocities)

    return Data(
        x=residual_velocities + affine_velocities if total_velocity else residual_velocities,
        pos=base_graph.x,
        affine_velocities=affine_velocities,
        edge_index=base_graph.edge_index,
        edge_attr=base_graph.edge_attr,
        box=base_graph.box if hasattr(base_graph, "box") else None,
        box_tensor=base_graph.box_tensor if hasattr(base_graph, "box_tensor") else None,
        t=base_graph.t if hasattr(base_graph, "t") else None,
    )


# Rollout generation
def get_rollout(
    input_graphs: list[Data],
    gnn_simulator: GNNModel,
    num_steps: int,
    gnn_history: int,
    barostat_config: dict | None,
    lj_params: LJInteractionParams | None,
    device: str = "cuda",
) -> list[Data]:
    num_particles = input_graphs[0].num_nodes

    r0 = input_graphs[0].edge_attr[:, -2]

    if barostat_config is None:
        mode = 'none'
    elif "x_only" in barostat_config:
        mode = "x_only"
    else:
        mode = "full"

    if mode == "full":
        C_coupling = barostat_config["C_coupling"]
        damping_coeff = barostat_config["damping"]
        target_pressure = barostat_config["target_pressure"]
        temperature = barostat_config["temperature"]
        dt = barostat_config["dt"]
        default_skip = barostat_config["default_skip"]
        stride_dt = default_skip * dt
        W_y = C_coupling * num_particles * (stride_dt**2)
        damping = damping_coeff * num_particles * stride_dt

    rollout = []
    for graph in input_graphs:
        rollout.append(graph)

    # Calculate box compression factor
    b0 = input_graphs[-2].box_tensor[0]
    b1 = input_graphs[-1].box_tensor[0]
    box_delta_x = b1 - b0

    # Estimate initial box velocity in Y axis
    if mode == "full":
        if gnn_history >= 2:
            current_box_vel_y = estimate_initial_box_vel_y_accurate(input_graphs[-3], input_graphs[-2], input_graphs[-1], stride_dt)
        else:
            current_box_vel_y = estimate_initial_box_vel_y(input_graphs[-2], input_graphs[-1], stride_dt)
    else:
        pass

    # build input graph
    input_graphs = [g.cpu() for g in rollout[-gnn_history - 1 :]]
    input_graph = build_velocity_graph_correction(input_graphs).to(device)

    inps = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)
    model_output = gnn_simulator(input_graph, is_training=False)
    predicted_graph = gnn_simulator.update(inps, model_output)

    new_box_tensor = predicted_graph.box_tensor.clone()
    match mode:
        # do not update box at all, a standard old rollout
        case "none": 
            pass
        # update box Lx (constant strain rate, exact value)
        case "x_only": 
            new_box_tensor[0] = new_box_tensor[0] + box_delta_x
        # update both Lx (constant strain rate) and Ly (barostat)
        case "full": 
            new_box_tensor[0] = new_box_tensor[0] + box_delta_x
            new_ly, new_vel_y = update_box_y_thermodynamic(
                positions=predicted_graph.pos,
                edge_index=inps.cur_graph.edge_index,
                edge_attr=inps.cur_graph.edge_attr,
                current_box=inps.cur_graph.box_tensor,
                r0=r0.to(predicted_graph.pos.device),
                box_vel_y=current_box_vel_y,
                W_y=W_y,
                damping=damping,
                stride_dt=stride_dt,
                lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                target_pressure=target_pressure,
                temperature=temperature,
            )
            new_box_tensor[1] = new_ly
            current_box_vel_y = new_vel_y

    # Update the box to graph
    predicted_graph.box_tensor = new_box_tensor

    # Recompute edge_attr depending on format
    function_output = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
    if isinstance(function_output, Tensor):
        predicted_graph.edge_attr = function_output
    
    elif isinstance(function_output, tuple):
        edge_index, edge_attr = function_output
        predicted_graph.edge_index = edge_index
        predicted_graph.edge_attr = edge_attr
    
    # predicted_graph.forces = compute_per_particle_forces(predicted_graph, r0=r0.to(predicted_graph.x.device), cutoff=lj_params.cutoff)
    rollout.append(predicted_graph.cpu().detach())

    # Rollout loop
    for _ in range(num_steps):
        input_graph = build_velocity_graph_correction([rollout[i].to(device) for i in range(-gnn_history - 1, 0, 1)]).to(device)

        model_output = gnn_simulator(input_graph, is_training=False)
        update_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)
        predicted_graph = gnn_simulator.update(update_inputs, model_output)

        # Update X
        new_box_tensor = predicted_graph.box_tensor.clone()
        match mode:

            # do not update box at all, a standard old rollout
            case "none": 
                pass

            # update box Lx (constant strain rate, exact value)
            case "x_only": 
                new_box_tensor[0] = new_box_tensor[0] + box_delta_x

            # update both Lx (constant strain rate) and Ly (barostat)
            case "full": 
                new_box_tensor[0] = new_box_tensor[0] + box_delta_x
                new_ly, new_vel_y = update_box_y_thermodynamic(
                    positions=predicted_graph.pos,
                    edge_index=update_inputs.cur_graph.edge_index,
                    edge_attr=update_inputs.cur_graph.edge_attr,
                    current_box=update_inputs.cur_graph.box_tensor,  # Use CURRENT box to calc pressure
                    r0=r0.to(predicted_graph.pos.device),
                    box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
                    W_y=W_y,
                    damping=damping,
                    stride_dt=stride_dt,
                    lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                    target_pressure=target_pressure,
                    temperature=temperature,
                )
                new_box_tensor[1] = new_ly
                current_box_vel_y = new_vel_y

        predicted_graph.box_tensor = new_box_tensor

        # Recompute edge_attr depending on format
        # predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
        function_output = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
        if isinstance(function_output, Tensor):
            predicted_graph.edge_attr = function_output
        
        elif isinstance(function_output, tuple):
            edge_index, edge_attr = function_output
            predicted_graph.edge_index = edge_index
            predicted_graph.edge_attr = edge_attr
        # predicted_graph.forces = compute_per_particle_forces(predicted_graph, r0=r0.to(predicted_graph.x.device), cutoff=lj_params.cutoff)
        rollout.append(predicted_graph.cpu().detach())

    return rollout


def simulate_then_rollout(
    starting_graph: Data,
    gnn_simulator: GNNModel,
    gnn_history: int,
    barostat_config: dict,
    lj_params: LJInteractionParams | None,
    md_steps: int,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:
    """Initial bootstrap trajectory is constructed via custom MD engine."""

    num_particles = starting_graph.num_nodes
    dt = barostat_config["dt"]
    default_skip = barostat_config["default_skip"]
    stride_dt = default_skip * dt
    W_y = barostat_config["C_coupling"] * num_particles * (stride_dt**2)
    damping = barostat_config["damping"] * num_particles * stride_dt

    # MD trajectory
    starting_graph = starting_graph.to(device)
    starting_graph = to_f64(starting_graph)
    r0 = starting_graph.edge_attr[:, -2]
    simulator: DifferentiableCompression64 = DifferentiableCompression64(
        starting_graph.num_nodes,
        lj_params=lj_params,
    )
    simulator_rollout, _conditions = simulator.run_simulator(
        initial_data=starting_graph,
        steps=md_steps,
        r0=r0,
        debug=False,
        device=device,
    )

    # GNN Simulator trajectory
    indices = [i * default_skip for i in range(gnn_history + 1)]
    input_graphs = [simulator_rollout[i] for i in indices]
    input_graphs = [to_f32(g) for g in input_graphs]
    for g in input_graphs:
        g.pos = g.x

    rollout = [g for g in input_graphs]

    b0 = input_graphs[-2].box_tensor[0]
    b1 = input_graphs[-1].box_tensor[0]
    box_delta_x = b1 - b0

    if gnn_history >= 2:
        current_box_vel_y = estimate_initial_box_vel_y_accurate(input_graphs[-3], input_graphs[-2], input_graphs[-1], stride_dt)
    else:
        current_box_vel_y = estimate_initial_box_vel_y(input_graphs[-2], input_graphs[-1], stride_dt)

    for _ in range(rollout_steps + 1):
        input_graph = build_velocity_graph_correction(rollout[-gnn_history - 1 :]).to(device)

        model_output = gnn_simulator(input_graph, is_training=False)
        update_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)
        predicted_graph = gnn_simulator.update(update_inputs, model_output)

        # Scale box X
        new_lx = predicted_graph.box_tensor[0] + box_delta_x

        # Update box Y
        new_ly, new_vel_y = update_box_y_thermodynamic(
            positions=predicted_graph.pos,
            edge_index=update_inputs.cur_graph.edge_index,
            edge_attr=update_inputs.cur_graph.edge_attr,
            current_box=update_inputs.cur_graph.box_tensor,  # Use CURRENT box to calc pressure
            r0=r0.float().to(predicted_graph.pos.device),
            box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
            W_y=W_y,
            damping=damping,
            stride_dt=default_skip * dt,
            lj_cutoff=lj_params.cutoff if lj_params is not None else None,
            target_pressure=barostat_config["target_pressure"],
            temperature=barostat_config["temperature"],
        )

        new_box_tensor = torch.stack([new_lx, new_ly])
        current_box_vel_y = new_vel_y

        # Apply to graph
        predicted_graph.box_tensor = new_box_tensor

        function_output = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
        if isinstance(function_output, Tensor):
            predicted_graph.edge_attr = function_output
        elif isinstance(function_output, tuple):
            edge_index, edge_attr = function_output
            predicted_graph.edge_index = edge_index
            predicted_graph.edge_attr = edge_attr


        rollout.append(predicted_graph)

    return rollout


def rollout_cascade(
    models: list[GNNModel],
    initial_state: Data,
    num_steps: int,
    barostat_config: dict,
    box_delta_x: float,
    device: str = "cuda"
) -> list[Data]:
    """
    A rollout function for simulator cascade.
    """

    # Setup Barostat Parameters
    num_particles = initial_state.num_nodes
    r0 = initial_state.edge_attr[:, -2]
    C_coupling = barostat_config["C_coupling"]
    damping_coeff = barostat_config["damping"]
    target_pressure = barostat_config["target_pressure"]
    temperature = barostat_config["temperature"]
    dt = barostat_config["dt"]
    default_skip = barostat_config["default_skip"]
    stride_dt = default_skip * dt
    
    W_y = C_coupling * num_particles * (stride_dt**2)
    damping_params = damping_coeff * num_particles * stride_dt

    # Trajectory initialization
    current_trajectory = [initial_state.to(device)]
    
    # We maintain the box velocity state across the rollout
    current_box_vel_y = 0.0 
    
    # Ensure all models are in eval mode
    for m in models:
        m.eval()


    with torch.no_grad():
        for step in range(num_steps):
            
            # Model selection
            history_len = len(current_trajectory)
            if history_len <= len(models):
                # Warmup phase: use the model corresponding to current history depth
                active_model_idx = history_len - 1
            else:
                # Stable phase: use the most advanced model
                active_model_idx = len(models) - 1
            
            active_model = models[active_model_idx]
            
            # Construct input graph
            input_graph = build_velocity_graph_correction(current_trajectory[-len(models)::], panic_at_positions=False).to(device)
            
            # Define "Previous" and "Current" frames for the ModelInputs wrapper
            # If we only have 1 frame (start), prev and curr are the same.
            prev_frame = current_trajectory[-2] if history_len > 1 else current_trajectory[-1]
            curr_frame = current_trajectory[-1]
            
            model_inputs = ModelInputs(prev_frame, curr_frame, None)

            # Forward
            pred_delta = active_model(input_graph, is_training=False)

            # Update next state
            next_step_pred = active_model.update(model_inputs, pred_delta, recalc_edges=False)

            # Update Box (Barostat)
            new_box_tensor = next_step_pred.box_tensor.clone()
            # Update Lx
            new_box_tensor[0] = new_box_tensor[0] + box_delta_x
            
            # Update Ly and and box velocity y
            new_ly, new_vel_y = update_box_y_thermodynamic(
                positions=next_step_pred.pos,
                edge_index=model_inputs.cur_graph.edge_index,
                edge_attr=model_inputs.cur_graph.edge_attr,
                current_box=model_inputs.cur_graph.box_tensor,
                r0=r0.float().to(next_step_pred.pos.device),
                box_vel_y=current_box_vel_y,
                W_y=W_y,
                damping=damping_params,
                stride_dt=stride_dt,
                target_pressure=target_pressure,
                temperature=temperature,
            )
            
            new_box_tensor[1] = new_ly
            current_box_vel_y = new_vel_y
            
            # Assign updated box to the prediction
            next_step_pred.box_tensor = new_box_tensor

            # Recompute edges
            next_step_pred.edge_attr = get_correct_edge_attr(next_step_pred, recompute_stiff=False, panic_at_nontensor_box=True)

            # Add to the rollout
            current_trajectory.append(next_step_pred.detach())

    return current_trajectory

# Rollout generation with ITPO
def compute_combined_physics_loss(graph: Data, r0: Tensor, lj_cutoff: float | None, target_Pyy: float = 0.0) -> tuple[Tensor]:
    # Potential energy
    U = compute_potential_energy(graph, r0=r0.to(graph.x.device))

    # Per-particle Fy 
    forces = compute_per_particle_forces(graph, r0=r0.to(graph.x.device), cutoff=lj_cutoff, panic_at_nontensor_box=True)
    force_y_mse = torch.mean(forces[:, 1].pow(2))

    # Virial pressure Pyy (should match target, which normally equals to 0.0)
    stress_tensor = compute_virial_stress(graph, r0=r0.to(graph.x.device), lj_cutoff=lj_cutoff)
    Pyy = stress_tensor[1]
    stress_y_mse = (Pyy - target_Pyy).pow(2)

    return U, force_y_mse, stress_y_mse


def physical_inference_step(
    model: GNNModel,
    input_graph: Data,
    model_inputs: ModelInputs,
    barostat_config: dict,
    box_delta_x: float,
    r0: Tensor,
    current_box_vel_y: Tensor,
    itpo_weights: ITPOWeights
) -> tuple[Data, Tensor]:

    # Get current and previous graph
    curr_graph = model_inputs.cur_graph
    prev_graph = model_inputs.prev_graph

    # Get all barostat-related parameters
    num_particles = input_graph.num_nodes
    dt = barostat_config["dt"]
    default_skip = barostat_config["default_skip"]
    stride_dt = default_skip * dt
    W_y = barostat_config["C_coupling"] * num_particles * (stride_dt**2)
    damping = barostat_config["damping"] * num_particles * stride_dt


    # Forward
    with torch.no_grad():
        a_nn = model(input_graph)
        a_nn = model.output_normalizer.inverse(a_nn)  # Real space accelerations

    # Setup optimization on the predicted acceleration
    a_refined = torch.nn.Parameter(a_nn.clone())

    # Initialize optimizer
    optimizer = torch.optim.Adam([a_refined], lr=itpo_weights.learning_rate)

    def closure():
        optimizer.zero_grad()

        # Differentiable Forward Euler Integration
        v_curr = curr_graph.x - prev_graph.x
        v_next = v_curr + a_refined
        x_next = curr_graph.x + v_next

        # Construct new Data object
        predicted_graph = Data(
            x=x_next,
            pos=x_next,
            edge_index=curr_graph.edge_index,
            edge_attr=curr_graph.edge_attr,
            box=curr_graph.box if hasattr(curr_graph, "box") else None,
            box_tensor=curr_graph.box_tensor if hasattr(curr_graph, "box_tensor") else None,
        )

        # Apply uniaxial compression
        compressed_lx = predicted_graph.box_tensor[0] + box_delta_x

        # Get new Ly from barostat
        compressed_ly, _new_vel_y = update_box_y_thermodynamic(
            positions=predicted_graph.pos,
            edge_index=model_inputs.cur_graph.edge_index,
            edge_attr=model_inputs.cur_graph.edge_attr,
            current_box=model_inputs.cur_graph.box_tensor,  # Use CURRENT box to calc pressure
            r0=r0.float().to(predicted_graph.pos.device),
            box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
            W_y=W_y,
            damping=damping,
            stride_dt=stride_dt,
            target_pressure=barostat_config["target_pressure"],
            temperature=barostat_config["temperature"],
        )

        new_box_tensor = torch.stack([compressed_lx, compressed_ly])

        # Update graph with a new box
        predicted_graph.box_tensor = new_box_tensor
        predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True)

        # Calculate Losses

        # Anchor Loss
        loss_anchor = torch.mean((a_refined - a_nn) ** 2)

        # Physics Loss
        energy_loss, force_loss, pressure_loss = compute_combined_physics_loss(predicted_graph, r0=r0, target_Pyy=0.0)
        loss_physics = (
            itpo_weights.lambda_energy * energy_loss + 
            itpo_weights.lambda_force * force_loss + 
            itpo_weights.lambda_pressure * pressure_loss
        )

        # Total loss
        total_loss = loss_anchor + loss_physics

        total_loss.backward()
        return total_loss

    # Optimization Loop
    for _ in range(itpo_weights.refinement_iterations):
        optimizer.step(closure)

    # Final Forward Euler step
    with torch.no_grad():
        
        # Final update
        v_curr = curr_graph.x - prev_graph.x
        v_next = v_curr + a_refined
        x_next = curr_graph.x + v_next

        # Construct final Data object
        predicted_graph = Data(
            x=x_next,
            pos=x_next,
            edge_index=curr_graph.edge_index,
            edge_attr=curr_graph.edge_attr,
            box=curr_graph.box if hasattr(curr_graph, "box") else None,
            box_tensor=curr_graph.box_tensor if hasattr(curr_graph, "box_tensor") else None,
        )

        # Apply uniaxial compression
        compressed_lx = predicted_graph.box_tensor[0] + box_delta_x

        # Get new Ly from barostat
        compressed_ly, new_vel_y = update_box_y_thermodynamic(
            positions=predicted_graph.pos,
            edge_index=model_inputs.cur_graph.edge_index,
            edge_attr=model_inputs.cur_graph.edge_attr,
            current_box=model_inputs.cur_graph.box_tensor,  # Use CURRENT box to calc pressure
            r0=r0.float().to(predicted_graph.pos.device),
            box_vel_y=current_box_vel_y,  # Use ESTIMATED velocity
            W_y=W_y,
            damping=damping,
            stride_dt=stride_dt,
            target_pressure=barostat_config["target_pressure"],
            temperature=barostat_config["temperature"],
        )

        new_box_tensor = torch.stack([compressed_lx, compressed_ly])

        # Update graph with a new box
        predicted_graph.box_tensor = new_box_tensor
        predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True)

    return predicted_graph.detach(), new_vel_y.detach()


def physical_inference_step_STE(
    model: GNNModel,
    input_graph: Data,
    model_inputs: ModelInputs,
    barostat_config: dict,
    box_delta_x: float,
    r0: Tensor,
    current_box_vel_y: Tensor,
    itpo_weights: ITPOWeights,
) -> tuple[Data, Tensor]:

    # Get current and previous graph
    curr_graph = model_inputs.cur_graph
    prev_graph = model_inputs.prev_graph

    # Get all barostat-related parameters
    num_particles = input_graph.num_nodes
    dt = barostat_config["dt"]
    default_skip = barostat_config["default_skip"]
    stride_dt = default_skip * dt
    W_y = barostat_config["C_coupling"] * num_particles * (stride_dt**2)
    damping = barostat_config["damping"] * num_particles * stride_dt

    # GNN predicts initial accelerations
    a_nn = model(input_graph)
    a_nn = model.output_normalizer.inverse(a_nn)  # Real space accelerations

    # Setup optimization on the ACCELERATION
    a_refined = torch.nn.Parameter(a_nn.detach().clone())
    optimizer = torch.optim.Adam([a_refined], lr=itpo_weights.learning_rate)

    # Detach all historical states for the inner loop
    cx_det = curr_graph.x.detach()
    px_det = prev_graph.x.detach()
    ea_det = curr_graph.edge_attr.detach()
    cb_det = (
        curr_graph.box_tensor.detach()
        if hasattr(curr_graph, "box_tensor") and curr_graph.box_tensor is not None
        else None
    )
    r0_det = r0.detach()

    # Handle current_box_vel_y whether it is a float or a Tensor
    vel_y_det = (
        current_box_vel_y.detach()
        if isinstance(current_box_vel_y, torch.Tensor)
        else current_box_vel_y
    )

    # Detach box and compression factor for the inner loop
    bcf_det = (
        box_delta_x.detach()
        if isinstance(box_delta_x, torch.Tensor)
        else box_delta_x
    )

    for _ in range(itpo_weights.refinement_iterations):
        # Enable gradients explicitly
        with torch.enable_grad():
            optimizer.zero_grad()

            v_curr = cx_det - px_det
            v_next = v_curr + a_refined
            x_next = cx_det + v_next

            # Construct temporary Data object
            predicted_graph = Data(
                x=x_next,
                pos=x_next,
                edge_index=curr_graph.edge_index,
                edge_attr=ea_det,
                box_tensor=cb_det,
            )

            # Apply uniaxial compression
            compressed_lx = predicted_graph.box_tensor[0] + bcf_det

            # Get new Ly from barostat
            compressed_ly, _temp_vel_y = update_box_y_thermodynamic(
                positions=predicted_graph.pos,
                edge_index=curr_graph.edge_index,
                edge_attr=ea_det,
                current_box=cb_det,
                r0=r0_det.float().to(predicted_graph.pos.device),
                box_vel_y=vel_y_det,
                W_y=W_y,
                damping=damping,
                stride_dt=stride_dt,
                target_pressure=barostat_config["target_pressure"],
                temperature=barostat_config["temperature"],
            )

            new_box_tensor = torch.stack([compressed_lx, compressed_ly])

            # Update graph with a new box
            predicted_graph.box_tensor = new_box_tensor
            predicted_graph.edge_attr = get_correct_edge_attr(
                predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True
            )

            # Calculate Losses
            loss_anchor = torch.mean((a_refined - a_nn.detach()) ** 2)
            energy_loss, force_loss, pressure_loss = compute_combined_physics_loss(
                predicted_graph, r0=r0_det, target_Pyy=0.0
            )

            loss_physics = (
                itpo_weights.lambda_energy * energy_loss
                + itpo_weights.lambda_force * force_loss
                + itpo_weights.lambda_pressure * pressure_loss
            )

            total_loss = loss_anchor + loss_physics
            total_loss.backward()

        optimizer.step()

    a_step = a_nn + (a_refined.detach() - a_nn).detach()

    v_curr = curr_graph.x - prev_graph.x
    v_next = v_curr + a_step
    x_next = curr_graph.x + v_next

    # Construct final Data object (Using original inputs to keep grad flowing)
    predicted_graph = Data(
        x=x_next,
        pos=x_next,
        edge_index=curr_graph.edge_index,
        edge_attr=curr_graph.edge_attr,
        box_tensor=curr_graph.box_tensor if hasattr(curr_graph, "box_tensor") else None,
    )

    # Apply uniaxial compression
    compressed_lx = predicted_graph.box_tensor[0] + bcf_det

    # Get new Ly from barostat
    compressed_ly, new_vel_y = update_box_y_thermodynamic(
        positions=predicted_graph.pos,
        edge_index=model_inputs.cur_graph.edge_index,
        edge_attr=model_inputs.cur_graph.edge_attr,
        current_box=model_inputs.cur_graph.box_tensor,
        r0=r0.float().to(predicted_graph.pos.device),
        box_vel_y=current_box_vel_y, 
        W_y=W_y,
        damping=damping,
        stride_dt=stride_dt,
        target_pressure=barostat_config["target_pressure"],
        temperature=barostat_config["temperature"],
    )

    new_box_tensor = torch.stack([compressed_lx, compressed_ly])

    # Update graph with a new box
    predicted_graph.box_tensor = new_box_tensor
    predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True)

    return predicted_graph, new_vel_y


def specialized_rollout(
    starting_graph: Data,
    gnn_simulator: GNNModel,
    gnn_history: int,
    barostat_config: dict,
    itpo_weights: ITPOWeights,
    md_steps: int,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:
    
    r0 = starting_graph.edge_attr[:, -2]
    default_skip = barostat_config["default_skip"]
    dt = barostat_config["dt"]
    stride_dt = default_skip * dt

    # Bootstrap with torch_simulator64
    starting_graph = to_f64(starting_graph).cuda()
    simulator: DifferentiableCompression64 = DifferentiableCompression64(starting_graph.num_nodes, factor_two=True, temp_langevin=0.0)
    simulator_rollout, _conditions = simulator.run_simulator(
        starting_graph,
        md_steps,
        debug=False,
        device=device,
        r0=r0.to(device),
    )

    # GNN Simulator trajectory
    indices = [i * default_skip for i in range(gnn_history + 1)]
    raw_input_graphs = [simulator_rollout[i] for i in indices]

    input_graphs = []
    for g in raw_input_graphs:
        
        # Convert to f32
        clean_g = to_f32(g)
        
        g.pos = g.x

        # Explicitly detach all tensors to kill the upstream graph
        clean_g.x = clean_g.x.detach()
        
        if hasattr(clean_g, 'pos') and clean_g.pos is not None:
            clean_g.pos = clean_g.pos.detach()
        else:
            clean_g.pos = clean_g.x 
            
        if hasattr(clean_g, 'box_tensor') and clean_g.box_tensor is not None:
            clean_g.box_tensor = clean_g.box_tensor.detach()
            
        if hasattr(clean_g, 'edge_attr') and clean_g.edge_attr is not None:
            clean_g.edge_attr = clean_g.edge_attr.detach()
            
        if hasattr(clean_g, 'edge_index') and clean_g.edge_index is not None:
            clean_g.edge_index = clean_g.edge_index.detach()
            
        input_graphs.append(clean_g)

    rollout = [g for g in input_graphs]

    b0 = input_graphs[-2].box_tensor[0]
    b1 = input_graphs[-1].box_tensor[0]
    box_delta_x = b1 - b0

    if gnn_history >= 2:
        current_box_vel_y = estimate_initial_box_vel_y_accurate(input_graphs[-3], input_graphs[-2], input_graphs[-1], stride_dt)
    else:
        current_box_vel_y = estimate_initial_box_vel_y(input_graphs[-2], input_graphs[-1], stride_dt)

    for _ in range(rollout_steps + 1):
        input_graph = build_velocity_graph_correction(rollout[-gnn_history - 1 :]).to(device)
        model_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)
        
        predicted_graph, current_box_vel_y = physical_inference_step(
            model=gnn_simulator,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0.float(),
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights,
        )

        rollout.append(predicted_graph)
    return rollout


def specialized_rollout_STE(
    starting_graph: Data,
    gnn_simulator: GNNModel,
    gnn_history: int,
    barostat_config: dict,
    itpo_weights: ITPOWeights,
    md_steps: int,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:

    r0 = starting_graph.edge_attr[:, -2]
    default_skip = barostat_config["default_skip"]
    dt = barostat_config["dt"]
    stride_dt = default_skip * dt

    # Bootstrap with torch_simulator64
    starting_graph = to_f64(starting_graph).to(device)
    simulator: DifferentiableCompression64 = DifferentiableCompression64(
        starting_graph.num_nodes, factor_two=True, temp_langevin=0.0
    )
    simulator_rollout, _conditions = simulator.run_simulator(
        starting_graph,
        md_steps,
        debug=False,
        device=device,
        r0=r0.to(device),
    )

    # GNN Simulator trajectory
    indices = [i * default_skip for i in range(gnn_history + 1)]
    raw_input_graphs = [simulator_rollout[i] for i in indices]

    input_graphs = []
    for g in raw_input_graphs:
        # 1. Convert to f32
        clean_g = to_f32(g)

        g.pos = g.x

        # # 2. THE FIREWALL: Explicitly detach all tensors to kill the upstream graph
        # clean_g.x = clean_g.x.detach()

        # if hasattr(clean_g, 'pos') and clean_g.pos is not None:
        #     clean_g.pos = clean_g.pos.detach()
        # else:
        #     clean_g.pos = clean_g.x

        # if hasattr(clean_g, 'box_tensor') and clean_g.box_tensor is not None:
        #     clean_g.box_tensor = clean_g.box_tensor.detach()

        # if hasattr(clean_g, 'edge_attr') and clean_g.edge_attr is not None:
        #     clean_g.edge_attr = clean_g.edge_attr.detach()

        # if hasattr(clean_g, 'edge_index') and clean_g.edge_index is not None:
        #     clean_g.edge_index = clean_g.edge_index.detach()

        input_graphs.append(clean_g)

    rollout = [g for g in input_graphs]

    b0 = input_graphs[-2].box_tensor[0]
    b1 = input_graphs[-1].box_tensor[0]
    box_delta_x = b1 - b0

    if gnn_history >= 2:
        current_box_vel_y = estimate_initial_box_vel_y_accurate(
            input_graphs[-3], input_graphs[-2], input_graphs[-1], stride_dt
        )
    else:
        current_box_vel_y = estimate_initial_box_vel_y(
            input_graphs[-2], input_graphs[-1], stride_dt
        )

    for _ in range(rollout_steps + 1):
        input_graph = build_velocity_graph_correction(rollout[-gnn_history - 1 :]).to(device)
        model_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)

        predicted_graph, current_box_vel_y = physical_inference_step_STE(
            model=gnn_simulator,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0,
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights,
        )

        rollout.append(predicted_graph)
    return rollout


def specialized_rollout_cascade(
    starting_graph: Data,
    gnn_models: list[GNNModel],
    barostat_config: dict,
    box_delta_x: float,
    itpo_weights: ITPOWeights,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:
    for m in gnn_models:
        m.eval()

    r0 = starting_graph.edge_attr[:, -2]

    rollout = [starting_graph.to(device)]
    current_box_vel_y = 0.0 
    
    for _ in range(rollout_steps + 1):

        # Pick active model based on existing frames
        history_len = len(rollout)
        if history_len <= len(gnn_models):
            # Warmup phase: use the model corresponding to current history depth
            active_model_idx = history_len - 1
        else:
            # Stable phase: use the most advanced model
            active_model_idx = len(gnn_models) - 1        
        active_model = gnn_models[active_model_idx]
        
        input_graph = build_velocity_graph_correction(
            rollout[-len(gnn_models)::],
            panic_at_positions=False,
            total_velocity=False,
        ).to(device)
        
        prev_graph = rollout[-2] if history_len > 1 else rollout[-1]
        curr_graph = rollout[-1]
        
        model_inputs = ModelInputs(
            prev_graph,
            curr_graph,
            None
        )
        
        predicted_graph, current_box_vel_y = physical_inference_step(
            model=active_model,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0,
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights
        )

        rollout.append(predicted_graph.detach())
    
    return rollout


# Calculate Poisson ratio
def calc_p_ratio_box_tensor(trajectory: list[Data], last_index: int = -1) -> Tensor:
    if hasattr(trajectory[0], "box_tensor") and isinstance(trajectory[0].box_tensor, Tensor):
        
        # Correct calculation
        strain_x = (trajectory[last_index].box_tensor[0] - trajectory[0].box_tensor[0]) / trajectory[0].box_tensor[0]
        strain_y = (trajectory[last_index].box_tensor[1] - trajectory[0].box_tensor[1]) / trajectory[0].box_tensor[1]
        return -strain_y / (strain_x + 1e-8)
        
        # Naive calculation (works because square initial box)
        dx = trajectory[last_index].box_tensor[0] - trajectory[0].box_tensor[0]
        dy = trajectory[last_index].box_tensor[1] - trajectory[0].box_tensor[1]
        return -dy / (dx + 1e-8)
    else:
        raise AttributeError("No box tensor.")


def calc_p_ratio_box(simulation: list[Data], index: int = -1) -> float:
    """Calculates Poisson ratio from the box data

    Parameters
    ----------
    simulation : list[Data]
        list of torch_geometric `Data` objects
    Returns
    -------
    float
        Poisson ratio
    """
    strain_x = (simulation[index].box.x - simulation[0].box.x) / simulation[0].box.x
    strain_y = (simulation[index].box.y - simulation[0].box.y) / simulation[0].box.y
    return -strain_y / strain_x


# Changing floating point precision
def to_f32(data: Data, optional_fields: list[str] | None = None) -> Data:
    if optional_fields is None:
        optional_fields = ["pos", "forces", "velocity", "angles", "acc", "box_tensor"]

    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()

    for field in optional_fields:
        if hasattr(data, field):
            attr = getattr(data, field)
            if isinstance(attr, Tensor):
                attr = attr.float()
                setattr(data, field, attr)
    data.dtype = torch.float32

    return data


def to_f64(data: Data, optional_fields: list[str] | None = None) -> Data:
    if optional_fields is None:
        optional_fields = ["pos", "forces", "velocity", "angles", "acc", "box_tensor", "box", "r0", "edge_attr"]

    data.x = data.x.double()
    data.edge_attr = data.edge_attr.double()

    for field in optional_fields:
        if hasattr(data, field):
            attr = getattr(data, field)
            if isinstance(attr, Tensor):
                attr = attr.double()
                setattr(data, field, attr)
    data.dtype = torch.float64

    return data
