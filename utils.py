import os
import subprocess
from itertools import batched
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree, scatter

from barostat_utils import estimate_initial_box_vel_y, estimate_initial_box_vel_y_accurate, update_box_y_thermodynamic
from pressure import compute_per_particle_forces
from training_utils import GNNModel, ModelInputs


# Functions dealing with datasets
def split_sims(data: list[list[Data]], segment_length: int, step_limit: int) -> list[list[Data]]:
    """Batch simulations into tuples of n graphs.
    Preserves some time relations but makes it more random for training."""

    new_data = []
    for sim in data:
        new_data += batched(sim[:step_limit], segment_length)
    return new_data


# Functions dealing with position graphs
def get_correct_edge_vec(graph: Data, panic_at_nontensor_box: bool = False) -> Tensor:
    pos = graph.x
    col = graph.edge_index[0]
    row = graph.edge_index[1]

    # 1. ENSURE BOX IS TENSOR
    # If we fall back to floats (else block), gradients for the box size DIE here.
    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, Tensor):
        box_size = graph.box_tensor
    else:
        if panic_at_nontensor_box:
            raise AttributeError("Graph does not have a tensor with box info.")
        else:
            box_size = torch.tensor([graph.box.x, graph.box.y], device=pos.device, dtype=pos.dtype)

    # 2. Raw displacement
    dr = pos[col] - pos[row]  # [E, 2]

    # Ensure box_size broadcasts correctly [1, 2] against dr [E, 2]
    box_tensor = box_size.view(1, 2)

    dr_corrected = dr - torch.round(dr / box_tensor) * box_tensor

    return dr_corrected


def get_correct_edge_attr(graph: Data, recompute_stiff: bool, panic_at_nontensor_box: bool = False) -> Tensor:
    """Compute correct edge attrbutes: edge vectors, edge lengths and bond stiffness."""

    # 1. Get Differentiable Vectors
    edge_vecs = get_correct_edge_vec(graph, panic_at_nontensor_box=panic_at_nontensor_box)

    # 2. Compute Norm
    edge_lengths = torch.norm(edge_vecs, dim=1)

    # 3. Handle Stiffness
    if recompute_stiff:
        # If optimizing stiffness, this path is active.
        stiff = 1.0 / edge_lengths
    else:
        # Note: If just optimizing positions, this passes the old constant stiffness.
        # Ensure we don't accidentally detach if stiffness was meant to be learned.
        stiff = graph.edge_attr[:, -1]

    # 4. Stack
    # Use column_stack or simple stack.
    # Result shape: [E, 4] -> (dx, dy, length, k)
    return torch.column_stack((edge_vecs, edge_lengths, stiff))


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
    periodic_edges: bool = False,
    box: bool = False,
    node_color: str = "skyblue",
    node_size: float = 20,
    node_labels: bool = False,
    chosen_nodes: Optional[list] = None,
    chosen_nodes_color: str = "red",
):
    plt.figure(figsize=(7, 7))

    graph = graph.cpu().detach()
    G = nx.Graph()

    # Add nodes
    for index, node in enumerate(graph.x):
        G.add_node(index)

    # Add edges
    if edges:
        edge_index = graph.edge_index.numpy().T
        for edge in edge_index:
            edge_len = torch.norm(graph.x[edge[1]] - graph.x[edge[0]])
            if not periodic_edges:
                if (edge_len > (torch.max(graph.x[:, 0]) - torch.min(graph.x[:, 0])) / 2) or (
                    edge_len > (torch.max(graph.x[:, 1]) - torch.min(graph.x[:, 1])) / 2
                ):
                    continue
                else:
                    G.add_edge(edge[0], edge[1])
            else:
                G.add_edge(edge[0], edge[1])

    pos = {index: (node[0].item(), node[1].item()) for index, node in enumerate(graph.x)}

    if box:
        B = nx.Graph()
        box_corners = [i for i in range(len(graph.x) + 1, len(graph.x) + 5)]
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

        nx.draw_networkx(B, b_pos, with_labels=False, node_color="black", node_size=1)

    # Create labels dictionary if node_labels is True
    labels = {i: str(i) for i in range(len(graph.x))} if node_labels else None

    # create nodelist and node_color
    nodelist = list(G)
    node_color = [node_color for i in range(len(nodelist))]
    if chosen_nodes:
        for node in chosen_nodes:
            node_color[node] = chosen_nodes_color

    nx.draw_networkx(
        G,
        pos,
        with_labels=node_labels,
        labels=labels,
        node_color=node_color,
        node_size=node_size,
        font_size=10,
    )

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
    chosen_nodes1: Optional[list] = None,
    chosen_nodes2: Optional[list] = None,
    chosen_nodes_color: str = "red",
    titles: Optional[List[str]] = None,
):
    """
    Plots two PyTorch Geometric graphs side-by-side using the specified visualization logic.
    """

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

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


def get_affine_residual_velocity(curr_graph: Data, target_graph: Data) -> Tuple[Tensor, Tensor]:
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


def build_velocity_graph_correction(input_graphs: List[Data], total_velocity: bool = False, panic_at_positions: bool = False) -> Data:
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
        raise Exception("Only one graph in input_graphs, can't construct velocity input.")

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


# One rollout function to rule them all
def get_rollout(
    input_graphs: List[Data],
    gnn_simulator: GNNModel,
    num_steps: int,
    gnn_history: int,
    barostat_config: dict | None,
    device: str = "cuda",
) -> List[Data]:
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
    box_compression_factor = b1 / b0

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
            new_box_tensor[0] = new_box_tensor[0] * box_compression_factor
        # update both Lx (constant strain rate) and Ly (barostat)
        case "full": 
            new_box_tensor[0] = new_box_tensor[0] * box_compression_factor
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
                target_pressure=target_pressure,
                temperature=temperature,
            )
            new_box_tensor[1] = new_ly
            current_box_vel_y = new_vel_y

    # Update the box to graph
    predicted_graph.box_tensor = new_box_tensor

    predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True)
    predicted_graph.forces = compute_per_particle_forces(predicted_graph, r0=r0.to(predicted_graph.x.device))
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
                new_box_tensor[0] = new_box_tensor[0] * box_compression_factor
            # update both Lx (constant strain rate) and Ly (barostat)
            case "full": 
                new_box_tensor[0] = new_box_tensor[0] * box_compression_factor
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
                    target_pressure=target_pressure,
                    temperature=temperature,
                )
                new_box_tensor[1] = new_ly
                current_box_vel_y = new_vel_y

        predicted_graph.box_tensor = new_box_tensor

        # Update edges again
        predicted_graph.edge_attr = get_correct_edge_attr(predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True)
        predicted_graph.forces = compute_per_particle_forces(predicted_graph, r0=r0.to(predicted_graph.x.device))
        rollout.append(predicted_graph.cpu().detach())

    return rollout


# Calculate Poisson ratio
def calc_p_ratio_box_tensor(trajectory: list[Data], last_index: int = -1) -> Tensor:
    if hasattr(trajectory[0], "box_tensor") and isinstance(trajectory[0].box_tensor, Tensor):
        denom = trajectory[last_index].box_tensor[0] - trajectory[0].box_tensor[0]
        num = trajectory[last_index].box_tensor[1] - trajectory[0].box_tensor[1]
        return -(num) / (denom + 1e-8)
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
    return -(simulation[index].box.y - simulation[0].box.y) / (simulation[index].box.x - simulation[0].box.x)


# Changing floating point precision
def to_f32(data: Data, optional_fields: list[str] = ["pos", "forces", "velocity", "angles", "acc", "box_tensor"]) -> Data:
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


def to_f64(data: Data, optional_fields: list[str] = ["pos", "forces", "velocity", "angles", "acc", "box_tensor"]) -> Data:
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
