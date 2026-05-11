from typing import List

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected


def filter_directed(edge_index: Tensor) -> Tensor:
    # Return edge index if its directed already
    if not is_undirected(edge_index):
        return torch.ones_like(edge_index[0]).bool()
    else:
        return edge_index[0] < edge_index[1]


def to_directed_graph(graph: Data) -> Data:

    edge_index = graph.edge_index
    edge_mask = filter_directed(edge_index)

    unique_edge_index = edge_index.T[edge_mask].T
    unique_edge_attr = graph.edge_attr[edge_mask]

    return Data(
        x=graph.x,
        edge_index=unique_edge_index,
        edge_attr=unique_edge_attr,
        box=graph.box,
    )


def to_undirected_graph(graph: Data, reduce: str = "mean") -> Data:
    """
    `reduce` -- ("add", "mean", "min", "max", "mul"). (default: "mean")
    """
    if is_undirected(graph.edge_index):
        return graph
    else:
        edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes, reduce=reduce)
        return Data(
            x=graph.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box=graph.box if hasattr(graph, "box") else None,
            time=graph.time if hasattr(graph, "time") else None,
        )


def compute_angle_indices(edge_index: Tensor) -> Tensor:
    """
    Finds all triplets (i, j, k) such that j is connected to i and k.
    Returns tensor of shape [3, Num_Angles]
    """
    # Convert edge_index to adjacency list format for fast lookup
    src, dst = edge_index
    idx = torch.argsort(src)
    src_sorted = src[idx]
    dst_sorted = dst[idx]

    triplets = []

    # Iterate through each node to find its neighbors
    unique_nodes, counts = torch.unique(src_sorted, return_counts=True)

    start_idx = 0
    for node, count in zip(unique_nodes, counts):
        if count < 2:
            start_idx += count
            continue

        # Get all neighbors of node 'j'
        neighbors = dst_sorted[start_idx : start_idx + count]

        # Create pairs of neighbors (i, k) for central node j
        n_indices = torch.combinations(neighbors, r=2)

        # Store as (i, j, k)
        # j is the center
        j_col = torch.full(
            (n_indices.shape[0], 1),
            node.item(),
            dtype=torch.long,
            device=edge_index.device,
        )

        # Result: [i, j, k]
        triplet_batch = torch.cat([n_indices[:, :1], j_col, n_indices[:, 1:]], dim=1)
        triplets.append(triplet_batch)

        start_idx += count

    if len(triplets) == 0:
        return torch.empty((3, 0), dtype=torch.long, device=edge_index.device)

    return torch.cat(triplets, dim=0).t()


def compute_angles(pos: Tensor, angle_index: Tensor, box_tensor: Tensor) -> Tensor:
    """
    Computes angle in radians for triplets.
    angle_index: [3, A] where rows are (i, j, k)
    """
    i, j, k = angle_index


    vec_ji = pos[i] - pos[j]
    vec_jk = pos[k] - pos[j]


    box = box_tensor.view(1, 2)

    vec_ji = vec_ji - torch.round(vec_ji / box) * box
    vec_jk = vec_jk - torch.round(vec_jk / box) * box


    dot_product = (vec_ji * vec_jk).sum(dim=1)

    norm_ji = torch.norm(vec_ji, dim=1) + 1e-12  # Epsilon for stability
    norm_jk = torch.norm(vec_jk, dim=1) + 1e-12

    cos_theta = dot_product / (norm_ji * norm_jk)
    cos_theta = torch.clamp(cos_theta, -0.99999, 0.99999)

    angles = torch.rad2deg(torch.acos(cos_theta))
    return angles


def prepare_traj(traj: List[Data], calc_angles: bool = True) -> List[Data]:
    bidirect = []
    for graph in traj:
        bidirect_graph = to_undirected_graph(graph)
        bidirect.append(bidirect_graph)

    triplet = compute_angle_indices(bidirect[0].edge_index)
    # r0 = bidirect[0].edge_attr[:, -2]
    for graph in bidirect:
        graph.box_tensor = torch.tensor(
            [graph.box.x, graph.box.y], dtype=torch.float32, device="cpu"
        )
        if calc_angles:
            angles = compute_angles(graph.x, triplet, graph.box_tensor)
            graph.angles = angles
        graph.angle_index = triplet
        # graph.forces = compute_per_particle_forces(graph, r0=r0)
        graph.pos = graph.x.clone()

    return bidirect
