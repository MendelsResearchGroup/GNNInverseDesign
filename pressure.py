import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected


def compute_potential_energy(graph: Data, r0: Tensor) -> Tensor:
    edge_attr = graph.edge_attr
    dist = edge_attr[:, -2]
    k = edge_attr[:, -1]

    energy_per_bond = k * (dist - r0).pow(2)

    total_potential_energy = torch.sum(energy_per_bond) / 2.0

    return total_potential_energy


def compute_virial_stress(graph: Data, r0: Tensor) -> Tensor:
    pos = graph.pos
    device = pos.device
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    box_size = graph.box_tensor
    k = edge_attr[:, -1]

    sender, receiver = edge_index
    r_vec = pos[sender] - pos[receiver]
    box_tensor = box_size.view(1, 2).to(device)
    r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor
    dist = torch.norm(r_vec, dim=1)

    # F = -2 * k * (r - r0)
    force_mag = -2.0 * k * (dist - r0)

    # F_vec = (r_vec / dist) * force_mag
    unit_vec = r_vec / (dist.unsqueeze(1) + 1e-12)
    force_vec = unit_vec * force_mag.unsqueeze(1)

    term_xx = force_vec[:, 0] * r_vec[:, 0]
    term_yy = force_vec[:, 1] * r_vec[:, 1]
    term_xy = force_vec[:, 0] * r_vec[:, 1]

    area = box_tensor[0, 0] * box_tensor[0, 1]
    # 0.5 is for undirected edges
    p_xx = 0.5 * torch.sum(term_xx) / area
    p_yy = 0.5 * torch.sum(term_yy) / area
    p_xy = 0.5 * torch.sum(term_xy) / area

    return torch.stack([p_xx, p_yy, p_xy])


def compute_total_stress(graph: Data, r0: Tensor, temperature: float = 1e-7):
    KB_METAL = 8.6173303e-5

    pos = graph.pos
    edge_index = graph.edge_index
    k = graph.edge_attr[:, -1]
    lx, ly = graph.box_tensor[0], graph.box_tensor[1]
    area = lx * ly

    row, col = edge_index
    r_vec = pos[row] - pos[col]
    r_vec[:, 0] -= lx * torch.round(r_vec[:, 0] / lx)
    r_vec[:, 1] -= ly * torch.round(r_vec[:, 1] / ly)
    dist = torch.norm(r_vec, dim=1) + 1e-12

    # Virial Components
    force_mag = -2.0 * k * (dist - r0)

    # Ratio used for virial: F_i * r_j
    # (force_mag / dist) * (r_i * r_j)
    ratio = force_mag / dist

    # 0.5 for bidirectional edges
    p_v_xx = 0.5 * torch.sum(ratio * r_vec[:, 0] ** 2) / area
    p_v_yy = 0.5 * torch.sum(ratio * r_vec[:, 1] ** 2) / area

    # Kinetic Term
    p_kinetic = (pos.shape[0] * KB_METAL * temperature) / area

    return torch.stack([p_v_xx + p_kinetic, p_v_yy + p_kinetic])


def compute_per_particle_forces(graph: Data, r0: Tensor, panic_at_nontensor_box: bool = True) -> Tensor:
    pos = graph.x

    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, Tensor):
        box_size = graph.box_tensor
    else:
        if panic_at_nontensor_box:
            raise AttributeError("Graph does not have a tensor with box info.")
        else:
            box_size = torch.tensor([graph.box.x, graph.box.y], device=pos.device, dtype=pos.dtype)

    edge_index = graph.edge_index
    stiffness = graph.edge_attr[:, -1]

    if is_undirected(edge_index):
        sources, targets = edge_index[0], edge_index[1]
        edge_stiffness = stiffness
    else:
        sources = torch.cat([edge_index[0], edge_index[1]])
        targets = torch.cat([edge_index[1], edge_index[0]])
        edge_stiffness = torch.cat([stiffness, stiffness])

    source_pos = pos[sources]
    target_pos = pos[targets]
    vecs = target_pos - source_pos

    half_box = box_size * 0.5
    vecs = torch.where(vecs > half_box, vecs - box_size, vecs)
    vecs = torch.where(vecs <= -half_box, vecs + box_size, vecs)

    lengths = torch.norm(vecs, dim=1)

    unit_vecs = vecs / lengths.unsqueeze(1)

    force_magnitudes = -edge_stiffness * (lengths - r0)
    force_vectors = force_magnitudes.unsqueeze(1) * unit_vecs

    forces = torch.zeros_like(pos).index_add(0, targets, force_vectors)

    return forces
