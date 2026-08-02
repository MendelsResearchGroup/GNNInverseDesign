from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected


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


# Handle edge attributes
@dataclass
class LJInteractionParams:
    epsilon: float = 0.01
    sigma: float = 1.0
    cutoff: float = 1.122

def get_correct_edge_vec(graph: Data, panic_at_nontensor_box: bool = False) -> Tensor:
    """Computes correct edge vecs based on particle positions in `data.x` 
    and adjecency matrix in `data.edge_index`.
    
    Only works for harmonic bond edges (old format).

    Parameters
    ----------
    graph : `Data`
        Input graph
    panic_at_nontensor_box : `bool`, optional
        by default False

    Returns
    -------
    Tensor
    """
    
    if graph.edge_attr.shape[1] != 4:
        raise ValueError("Only works for old format edges: [vx, vy, length, stiffness]")

    pos = graph.x
    col = graph.edge_index[0]
    row = graph.edge_index[1]

    # Ensure we have box as tensor
    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, Tensor):
        box_size = graph.box_tensor
    else:
        if panic_at_nontensor_box:
            raise AttributeError("Graph does not have a tensor with box info.")
        else:
            box_size = torch.tensor([graph.box.x, graph.box.y], device=pos.device, dtype=pos.dtype)

    # Raw displacement
    dr = pos[col] - pos[row]  # [E, 2]

    # Ensure box_size broadcasts correctly [1, 2] against dr [E, 2]
    box_tensor = box_size.view(1, 2)

    dr_corrected = dr - torch.round(dr / box_tensor) * box_tensor

    return dr_corrected


def get_correct_harmonic_edge_attr(data: Data, recompute_stiff: bool = False) -> Data:
    """
    Recomputes the harmonic bond edges in the new format:
    [is_orig, is_lj, vec_x, vec_y, current_bond_dist, bond_stiffness, original_r0]

    Expects the graph to also have the new format.

    `data` should have:
      - `data.x` contains the (N, 2) node coordinates.
      - `data.box_tensor` contains the (2,) box dimensions [Lx, Ly].
      - `data.edge_index` contains the adjacency matrix.
      - `data.edge_attr` contains the original features in order.
    """
    # Make sure we're working with the new format graph
    if data.edge_attr.shape[1] != 7:
        raise ValueError("Only works for new format edges:\n[is_harmonic, is_lj, vx, vy, dist_or_lengths, epsilon_or_stiff, sigma_or_r0]")

    # Recompute edge_vecs from scratch.
    # We use the flag in the edge_attr to understand which edges 
    # represent harmonic bonds.
    pos = data.x
    harmonic_mask = data.edge_attr[:, 0].bool()
    harmonic_edges = data.edge_attr[harmonic_mask]
    harmonic_edge_index = data.edge_index.T[harmonic_mask].T # Should be Shape [2, E_ij]

    col, row = harmonic_edge_index

    # Raw displacement
    dr = pos[col] - pos[row]  # [E, 2]

    # Ensure box_size broadcasts correctly [1, 2] against dr [E, 2]
    box_tensor = data.box_tensor.view(1, 2)

    # New harmonic edge vectors
    edge_vecs = dr - torch.round(dr / box_tensor) * box_tensor # Should be Shape [E_ij, 2]

    # Compute edge lengths
    edge_lengths = torch.linalg.vector_norm(edge_vecs, dim=1, keepdim=True)

    # Get old stiffness or recompute
    stiff = 1.0 / edge_lengths if recompute_stiff else harmonic_edges[:, 5:6]

    # Assemble everything back
    orig_one_hot = harmonic_edges[:, 0:2] # One-hot vectors do not change
    orig_r0 = harmonic_edges[:, 6:7]
    harmonic_edge_attr = torch.cat([orig_one_hot, edge_vecs, edge_lengths, stiff, orig_r0], dim=-1)

    return harmonic_edge_index, harmonic_edge_attr


def get_correct_lj_edge_attr(
    data: Data,
    lj_params: LJInteractionParams
) -> Data:
    """
    Computes edges for Lennard-Jones interactions dynamically with PBC.
    The cutoff is hardcoded to be 150% of the provided value in `lj_params`.
    Return bidirectional `edge_index` and `edge_attr`.
    
    `data` should have:
      - `data.x` contains the (N, 2) node coordinates.
      - `data.box_tensor` contains the (2,) box dimensions [Lx, Ly].
    """
    device = data.x.device
    box = data.box_tensor

    # Compute Pairwise Distances with PBC (Minimum Image Convention)
    # Using broadcasting: pos is (N, 2) -> diff is (N, N, 2)
    dr = data.x.unsqueeze(1) - data.x.unsqueeze(0)
    dr = dr - torch.round(dr / box) * box
    
    dist = torch.norm(dr, dim=-1) # Shape (N, N)

    # Filter by Cutoff and exclude self-loops (dist > 1e-6)
    buffered_cutoff = lj_params.cutoff * 1.5
    mask = (dist < buffered_cutoff) & (dist > 1e-6)

    # Extract LJ edge indices
    lj_edge_index = mask.nonzero(as_tuple=False).t().contiguous() # Shape: (2, E_lj)
    row, col = lj_edge_index
    num_lj_edges = lj_edge_index.shape[1]

    # Extract LJ Features
    # Target feature layout: [is_orig, is_lj, vec_x, vec_y, dist, epsilon, sigma]
    lj_one_hot = torch.tensor([[0.0, 1.0]], device=device).repeat(num_lj_edges, 1)
    lj_vec = dr[row, col]                                         # (E_lj, 2)
    lj_dist = dist[row, col].unsqueeze(-1)                        # (E_lj, 1)
    lj_eps = torch.full((num_lj_edges, 1), lj_params.epsilon, device=device)
    lj_sig = torch.full((num_lj_edges, 1), lj_params.sigma, device=device)

    lj_edge_attr = torch.cat([lj_one_hot, lj_vec, lj_dist, lj_eps, lj_sig], dim=-1)

    return lj_edge_index, lj_edge_attr


def get_correct_edge_attr(graph: Data, recompute_stiff: bool, lj_params: LJInteractionParams | None, panic_at_nontensor_box: bool = False) -> Tensor | tuple[Tensor, Tensor]:

    # Get box_tensor and edge_attr shape
    num_features = graph.edge_attr.shape[1]
    
    if num_features == 4: # No LJ, 4 edge features
        
        # Compute edge vectors
        edge_vecs = get_correct_edge_vec(graph, panic_at_nontensor_box=panic_at_nontensor_box)

        # Compute edge lengths
        edge_lengths = torch.linalg.vector_norm(edge_vecs, dim=1)
        
        # Get old stiffness or recompute
        stiff = 1.0 / edge_lengths if recompute_stiff else graph.edge_attr[:, 3]

        return torch.column_stack((edge_vecs, edge_lengths, stiff))

    elif num_features == 7: # With LJ, 7 edge features

        harmonic_edge_index, harmonic_edge_attr = get_correct_harmonic_edge_attr(graph, recompute_stiff=recompute_stiff)
        lj_edge_index, lj_edge_attr = get_correct_lj_edge_attr(graph, lj_params=lj_params)

        # Stack the two together 
        full_edge_index = torch.cat([harmonic_edge_index, lj_edge_index], dim=-1)
        full_edge_attr = torch.cat([harmonic_edge_attr, lj_edge_attr], dim=0)

        return full_edge_index, full_edge_attr 


def append_lj_interactions(
    data: Data,
    harmonic_r0: Tensor, 
    lj_params: LJInteractionParams
) -> Data:
    """
    Computes Lennard-Jones edges dynamically with PBC 
    and appends them to the graph. Works with old format graphs, use only once for data preparation! 
    
    `data` should contain:
      - `data.x` contains the (N, 2) node coordinates.
      - `data.box_tensor` contains the (2,) box dimensions [Lx, Ly].
      - `data.edge_index` contains the original harmonic bonds.
      - `data.edge_attr` contains the original features in order: 
        [edge_vec_x, edge_vec_y, stiffness, length].
    """
    device = data.x.device
    box = data.box_tensor

    # 1. Compute Pairwise Distances with PBC (Minimum Image Convention)
    # Using broadcasting: pos is (N, 2) -> diff is (N, N, 2)
    diff = data.x.unsqueeze(1) - data.x.unsqueeze(0)
    diff = diff - box * torch.round(diff / box)
    
    dist = torch.norm(diff, dim=-1) # Shape (N, N)

    # 2. Filter by Cutoff and exclude self-loops (dist > 1e-6)
    buffered_cutoff = lj_params.cutoff * 1.5
    mask = (dist < buffered_cutoff) & (dist > 1e-6)
    
    # Extract LJ edge indices
    lj_edge_index = mask.nonzero(as_tuple=False).t().contiguous() # Shape: (2, E_lj)
    row, col = lj_edge_index
    num_lj_edges = lj_edge_index.shape[1]

    # Extract LJ Features
    # Target feature layout: [is_orig, is_lj, vec_x, vec_y, dist, epsilon, sigma]
    lj_one_hot = torch.tensor([[0.0, 1.0]], device=device).repeat(num_lj_edges, 1)
    lj_vec = diff[row, col]                                         # (E_lj, 2)
    lj_dist = dist[row, col].unsqueeze(-1)                          # (E_lj, 1)
    lj_eps = torch.full((num_lj_edges, 1), lj_params.epsilon, device=device)
    lj_sig = torch.full((num_lj_edges, 1), lj_params.sigma, device=device)

    lj_edge_attr = torch.cat([lj_one_hot, lj_vec, lj_dist, lj_eps, lj_sig], dim=-1)

    # Update Original Edge Features
    orig_edge_index = data.edge_index
    num_orig_edges = orig_edge_index.shape[1]
    
    # We assume orig_edge_attr is [vec_x, vec_y, length, stiffness]
    # We need to map it to the same 7D layout.
    orig_one_hot = torch.tensor([[1.0, 0.0]], device=device).repeat(num_orig_edges, 1)
    orig_vec = data.edge_attr[:, 0:2]
    orig_length = data.edge_attr[:, 2:3]
    orig_stiffness = data.edge_attr[:, 3:4]
    
    # Final feature layout:
    # [is_orig, is_lj, vec_x, vec_y, current_bond_dist, bond_stiffness, original_r0]
    orig_edge_attr = torch.cat(
        [orig_one_hot, orig_vec, orig_length, orig_stiffness, harmonic_r0], 
        dim=-1
    )

    # Concatenate and Update the Graph
    data.edge_index = torch.cat([orig_edge_index, lj_edge_index], dim=-1)
    data.edge_attr = torch.cat([orig_edge_attr, lj_edge_attr], dim=0)

    return data


def physics_to_undirected(edge_index: Tensor, edge_attr: Tensor, num_nodes: int | None = None):
    """
    Converts a directed physics graph to an undirected graph, correctly 
    inverting spatial edge vectors for reverse edges and preserving scalars.
    """
    row, col = edge_index
    
    # 1. Create reverse edges
    rev_edge_index = torch.stack([col, row], dim=0)
    
    # 2. Clone attributes and invert the spatial vectors based on format
    rev_edge_attr = edge_attr.clone()
    num_features = edge_attr.shape[1]
    
    if num_features == 4 or num_features == 5:
        # Format: [vx, vy, length, stiffness]
        rev_edge_attr[:, 0:2] = -rev_edge_attr[:, 0:2]
    elif num_features == 7:
        # Format: [is_orig, is_lj, vx, vy, length_or_dist, stiffness_or_epsilon, zero_or_sigma]
        rev_edge_attr[:, 2:4] = -rev_edge_attr[:, 2:4]
    else:
        raise ValueError(f"Unexpected edge_attr dimension: {num_features}")
        
    # 3. Concatenate forward and reverse edges
    new_edge_index = torch.cat([edge_index, rev_edge_index], dim=-1)
    new_edge_attr = torch.cat([edge_attr, rev_edge_attr], dim=0)
    
    # 4. Coalesce to remove duplicates using 'mean'
    # 'mean' ensures that stiffness, length, sigma, and one-hot vectors 
    # remain exactly the same (e.g., (1.0 + 1.0) / 2 = 1.0).
    return coalesce(new_edge_index, new_edge_attr, num_nodes=num_nodes, reduce='mean')


def to_undirected_graph(graph: Data, reduce: str = "mean") -> Data:
    """
    `reduce` -- ("add", "mean", "min", "max", "mul"). (default: "mean")
    """
    if is_undirected(graph.edge_index):
        return graph
    else:
        edge_index, edge_attr = physics_to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
        return Data(
            x=graph.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            box=graph.box if hasattr(graph, "box") else None,
            box_tensor=graph.box_tensor if hasattr(graph, "box_tensor") else None,
            time=graph.time if hasattr(graph, "time") else None,
        )


def compute_angle_indices(edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
    """Finds all triplets (i, j, k) such that j is connected to i and k.
    Filters out Lennard-Jones edges if edge_attr is provided and in the 7D format.


    Returns
    -------
    Tensor
        Angle triplets of shape [3, Num_Angles]
    """
    
    # Filter out LJ edges if we have the 7D feature format
    if edge_attr is not None and edge_attr.shape[1] == 7:
        is_harmonic = edge_attr[:, 0].bool()
        edge_index = edge_index[:, is_harmonic]

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


def add_box_tensor(graph: Data) -> Data:
    device = graph.x.device
    graph.box_tensor = torch.tensor([graph.box.x, graph.box.y], device=device)
    return graph


def prepare_traj(traj: list[Data], lj_params: LJInteractionParams | None = None, rest_lengths: Tensor | None = None, calc_angles: bool = True) -> list[Data]:
    bidirect = []
    triplet = None

    for i, graph in enumerate(traj):

        # Make sure the graph is old format
        assert graph.edge_attr.shape[1] == 4

        # Add box_tensor attribute to the directed old format graph
        graph: Data = add_box_tensor(graph)

        if lj_params is None:
            # Make original old format graph bidirected, no LJ edges at this point
            bidirect_graph = to_undirected_graph(graph)

            # Grab the pure harmonic topology on step 0 BEFORE adding LJ
            if i == 0:
                triplet = compute_angle_indices(bidirect_graph.edge_index)

        # when LJ interactions are present, the equlibrium distance, rest length and stiffness are three independent variables.
        elif lj_params is not None: # Add LJ interactions if requested

            if rest_lengths is not None:
                # Add rest bond lengths as an additional edge feature to make it bidirect
                graph.edge_attr = torch.column_stack((graph.edge_attr, rest_lengths))            

                # Make graph bidirectional
                bidirect_graph = to_undirected_graph(graph)

                if i == 0:
                    triplet = compute_angle_indices(bidirect_graph.edge_index)

                # Exract bidirect rest lengths
                r0 = bidirect_graph.edge_attr[:, 4:5] # last column

                # Go back to 4 edge features
                graph.edge_attr = graph.edge_attr[:, :4]

            else: # fallback to the previous logic if no rest lengths available, but can be wrong.
                # Make graph bidirectional
                bidirect_graph = to_undirected_graph(graph)

                if i == 0:
                    triplet = compute_angle_indices(bidirect_graph.edge_index)
                    r0 = 1 / bidirect_graph.edge_attr[:, 3:4]

                else: # Otherwise, take rest bond lengths from step 0 that we added before
                    is_harmonic = bidirect[0].edge_attr[:, 0].bool()
                    harmonic_edges = bidirect[0].edge_attr[is_harmonic]
                    r0 = harmonic_edges[:, 6:7]
                    r0_other = harmonic_edges[:, 4:5]
                    assert (r0-r0_other).sum().abs() >= 1e-6 # sanity check, they SHOULD NOT BE EQUAL

            # LJ edges come bidirectional by design
            bidirect_graph = append_lj_interactions(bidirect_graph, r0, lj_params)
        
        # Append
        bidirect.append(bidirect_graph)

    for graph in bidirect:
        # Calculate angles if asked
        if calc_angles:
            angles = compute_angles(graph.x, triplet, graph.box_tensor)
            graph.angles = angles

        # Add angle index nonetheless
        graph.angle_index = triplet

        # Copy positions into a separate attribute
        graph.pos = graph.x.clone()

    return bidirect
