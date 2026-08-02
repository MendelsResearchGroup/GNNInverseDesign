import torch
from torch import Tensor
from torch_geometric.data import Data


def compute_potential_energy(graph: Data, r0: Tensor, lj_cutoff: float = 1.122) -> Tensor:
    edge_attr = graph.edge_attr
    num_features = edge_attr.shape[1]
    
    if num_features == 4: # old format
        dist = edge_attr[:, -2]
        k = edge_attr[:, -1]

        energy_harmonic = k * (dist - r0).pow(2)
        total_potential_energy = torch.sum(energy_harmonic) / 2.0

        return total_potential_energy

    elif num_features == 7: # new format
        is_harmonic = edge_attr[:, 0].bool()

        # Harmonic bonds parameters
        dist = edge_attr[:, 4]
        k = edge_attr[:, 5]
        r0_attr = edge_attr[:, 6]
        
        # LJ interactions parameters
        epsilon = edge_attr[:, 5]
        sigma = edge_attr[:, 6]

        # Harmonic Energy
        energy_harmonic = k * (dist - r0_attr).pow(2)

        # Lennard-Jones Energy
        safe_dist = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
        sr6 = (sigma / safe_dist) ** 6
        sr12 = sr6 ** 2
        
        energy_LJ = 4.0 * epsilon * (sr12 - sr6) + epsilon
        energy_LJ = torch.where(dist < lj_cutoff, energy_LJ, torch.zeros_like(energy_LJ))

        # Pick correct based on what interaction the edge corresponds to
        edge_energies = torch.where(is_harmonic, energy_harmonic, energy_LJ)

        # Divide by 2 because bidirectional graph
        total_potential_energy = torch.sum(edge_energies) / 2.0

        return total_potential_energy


def compute_virial_stress(graph: Data, r0: Tensor, lj_cutoff: float | None = 1.122) -> Tensor:
    pos = graph.pos
    device = pos.device
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    box_size = graph.box_tensor

    # Compute edge vecs
    sender, receiver = edge_index
    r_vec = pos[sender] - pos[receiver]
    box_tensor = box_size.view(1, 2).to(device)
    r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor

    # Compute bond lengths / LJ interaction distance
    dist = torch.norm(r_vec, dim=1)

    num_features = edge_attr.shape[1]
    if num_features == 4: # old format

        k = edge_attr[:, 3]

        # F = -2 * k * (r - r0)
        force_mag = -2.0 * k * (dist - r0)

    elif num_features == 7: # new format
                
        is_harmonic = edge_attr[:, 0].bool()
        
        k = edge_attr[:, 5]
        epsilon = edge_attr[:, 5]
        
        r0 = edge_attr[:, 6] # new format r0 is always stored as edge_attr[:, 6]
        sigma = edge_attr[:, 6]

        # Harmonic Force
        force_harmonic = -2.0 * k * (dist - r0)

        # Lennard-Jones Force
        safe_dist = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
        sr6 = (sigma / safe_dist) ** 6
        sr12 = sr6 ** 2
        force_lj = (24.0 * epsilon / safe_dist) * (2.0 * sr12 - sr6)
        force_lj = torch.where(dist < lj_cutoff, force_lj, torch.zeros_like(force_lj))
        
        force_mag = torch.where(is_harmonic, force_harmonic, force_lj)
    else:
        raise ValueError(f"Unexpected edge_attr dimension: {num_features}")

    # Compute stress tensor terms
    unit_vec = r_vec / (dist.unsqueeze(1) + 1e-12)
    force_vec = unit_vec * force_mag.unsqueeze(1)

    term_xx = force_vec[:, 0] * r_vec[:, 0]
    term_yy = force_vec[:, 1] * r_vec[:, 1]
    term_xy = force_vec[:, 0] * r_vec[:, 1]

    area = box_size[0] * box_size[1]

    # Virial Stress Tensor 
    total_stress = torch.stack([
        0.5 * torch.sum(term_xx) / area,
        0.5 * torch.sum(term_yy) / area,
        0.5 * torch.sum(term_xy) / area
    ])

    return total_stress


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


def compute_per_particle_forces(graph: Data, r0: Tensor | None, cutoff: float | None = 1.122, panic_at_nontensor_box: bool = True) -> Tensor:
    """Computes forces for both old and new format.

    Parameters
    ----------
    graph : Data

    r0 : Optional[Tensor]
        r0 is only used for old format calculation
    panic_at_nontensor_box : bool, optional
        raise error if `box_tensor` attribute is missing from `graph`

    Returns
    -------
    Tensor
        Per-particle forces tensor [N, 2]

    Raises
    ------
    AttributeError
        when `box_tensor` attribute is missing from `graph`
    ValueError
        number of edge features is not 4 or 7
    """
    pos = graph.x

    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, Tensor):
        box_size = graph.box_tensor
    else:
        if panic_at_nontensor_box:
            raise AttributeError("Graph does not have a tensor with box info.")
        else:
            box_size = torch.tensor([graph.box.x, graph.box.y], device=pos.device, dtype=pos.dtype)

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    box_tensor = box_size.view(1, 2)

    sources, targets = edge_index[0], edge_index[1]

    # Compute edge vecs with PBC
    source_pos = pos[sources]
    target_pos = pos[targets]
    vecs = target_pos - source_pos
    vecs = vecs - torch.round(vecs/box_tensor) * box_tensor

    # Compute edge lengths
    lengths = torch.linalg.vector_norm(vecs, dim=1)
    
    # Protect against perfectly overlapping particles to avoid NaN unit vectors
    safe_lengths = torch.where(lengths < 1e-6, torch.ones_like(lengths), lengths)
    unit_vecs = vecs / safe_lengths.unsqueeze(1)

    num_features = edge_attr.shape[1]

    if num_features == 4: # Old format with 4 edge feautures [vx, vy, length, stiffness]
        stiffness = edge_attr[:, -1]
        force_magnitudes = -2.0 * stiffness * (lengths - r0)

    elif num_features == 7: # New format with 7 edge features
        is_harmonic = edge_attr[:, 0].bool()

        # harmonic bonds parameters
        k = edge_attr[:, 5]
        r0 = edge_attr[:, 6]
        
        # LJ interactions parameters
        epsilon = edge_attr[:, 5]
        sigma = edge_attr[:, 6]

        # Harmonic Forces
        f_harm = -2.0 * k * (lengths - r0)

        # Lennard-Jones Forces
        sr6 = (sigma / safe_lengths) ** 6
        sr12 = sr6 ** 2
        f_lj = (24.0 * epsilon / safe_lengths) * (2.0 * sr12 - sr6)
        f_lj = torch.where(lengths < cutoff, f_lj, torch.zeros_like(f_lj))
        
        # 3. Apply routing
        force_magnitudes = torch.where(is_harmonic, f_harm, f_lj)

    else:
        raise ValueError(f"Unexpected edge_attr dimension: {num_features}")

    force_vectors = force_magnitudes.unsqueeze(1) * unit_vecs

    forces = torch.zeros_like(pos).index_add(0, targets, force_vectors)

    return forces


def compute_stress_pressure(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    box_tensor: torch.Tensor,
    lj_cutoff: float | None = 1.122,
    r0_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    sender, receiver = edge_index
    
    # Minimum image convention
    r_vec = pos[sender] - pos[receiver]
    box_view = box_tensor.view(1, 2)
    r_vec = r_vec - torch.round(r_vec / box_view) * box_view
    dist = torch.norm(r_vec, dim=1)

    num_edge_features = edge_attr.shape[1]

    if num_edge_features == 4: # Old format: [vx, vy, length, stiffness]

        if r0_override is None: # For old format, r0 must be provided
            raise ValueError("For old format (edge_attr = [vx, vy, length, stiffness]), r0 must be explicitly given.")

        k = edge_attr[:, 3]
        r0 = r0_override
        force_mag = -2.0 * k * (dist - r0)

    elif num_edge_features == 7: # New Format: [is_orig, is_lj, vx, vy, dist, stiffness_or_eps, r0_or_sigma]
        
        is_orig = edge_attr[:, 0].bool()
        
        k = edge_attr[:, 5]
        epsilon = edge_attr[:, 5]
        
        r0 = edge_attr[:, 6] # new format r0 is always stored as edge_attr[:, 6]
        sigma = edge_attr[:, 6]

        # Harmonic Force
        force_harmonic = -2.0 * k * (dist - r0)

        # Lennard-Jones Force
        safe_dist = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
        sr6 = (sigma / safe_dist) ** 6
        sr12 = sr6 ** 2
        force_lj = (24.0 * epsilon / safe_dist) * (2.0 * sr12 - sr6)
        force_lj = torch.where(dist < lj_cutoff, force_lj, torch.zeros_like(force_lj))
        
        force_mag = torch.where(is_orig, force_harmonic, force_lj)
    else:
        raise ValueError(f"Unexpected edge_attr dimension: {num_edge_features}")

    # Compute stress tensor terms
    unit_vec = r_vec / (dist.unsqueeze(1) + 1e-12)
    force_vec = unit_vec * force_mag.unsqueeze(1)

    term_xx = force_vec[:, 0] * r_vec[:, 0]
    term_yy = force_vec[:, 1] * r_vec[:, 1]
    term_xy = force_vec[:, 0] * r_vec[:, 1]

    # 2D network actually has 3D box: zlo = -0.1, zhi = +0.1
    volume = box_tensor[0] * box_tensor[1] * 0.2

    # Virial Stress Tensor 
    total_stress = torch.stack([
        0.5 * torch.sum(term_xx) / volume,
        0.5 * torch.sum(term_yy) / volume,
        0.5 * torch.sum(term_xy) / volume
    ])

    # Total Pressure (assuming 2D isotropic)
    total_pressure = torch.abs(total_stress[0] + total_stress[1]) / 2.0

    return total_stress, total_pressure


def compute_stress_curve(trajectory: list[Data], lj_cutoff: float | None = 1.122, sample_stride: int = 100, device: str = "cpu"):
    curve = []
    for i in range(0, len(trajectory), sample_stride):
        step_graph = trajectory[i]
        step_stress, _step_pressure = compute_stress_pressure(
            pos=step_graph.x,
            edge_index=step_graph.edge_index,
            edge_attr=step_graph.edge_attr,
            box_tensor=step_graph.box_tensor,
            lj_cutoff=lj_cutoff,
            r0_override=trajectory[0].edge_attr[:, -2], # this only matters for old format
        )
        curve.append(step_stress.to(device))
    return torch.stack(curve)
