import torch
from torch import Tensor
from torch_geometric.data import Data


def update_box_y_thermodynamic(
    positions: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    current_box: Tensor,
    r0: Tensor,
    box_vel_y: float,
    W_y: float,
    damping: float,
    stride_dt: float,
    lj_cutoff: float | None = None,
    target_pressure: float = 0.0,
    temperature: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    
    # LAMMPS simulation was done in metal units
    KB_METAL = 8.6173303e-5  # Boltzmann constant in eV/K

    row, col = edge_index

    lx = current_box[0] if current_box.dim() == 0 else current_box[0].item()
    ly = current_box[1] if current_box.dim() == 0 else current_box[1].item()
    volume = lx * ly
    num_particles = positions.shape[0]

    # Calculate PBC (Minimum Image Convention)
    dy = positions[row, 1] - positions[col, 1]
    dy = dy - ly * torch.round(dy / ly)

    dx = positions[row, 0] - positions[col, 0]
    dx = dx - lx * torch.round(dx / lx)

    dist = torch.norm(torch.stack([dx, dy], dim=1), dim=1)

    num_features = edge_attr.shape[1]

    if num_features == 4: # Old format of edge features [vs, vy, length, stiffness]
        stiffness = edge_attr[:, -1] # stiffness should always be edge_attr[:, -1]
        device = stiffness.device

        # Force calculation: -2.0 * k * (r - r0)
        force_mag = -2.0 * stiffness * (dist - r0.to(device))

    elif num_features == 7: # New format
        is_harmonic = edge_attr[:, 0].bool()

        # harmonic bonds parameters
        k = edge_attr[:, 5]
        r0 = edge_attr[:, 6]
        
        # LJ interactions parameters
        epsilon = edge_attr[:, 5]
        sigma = edge_attr[:, 6]

        # Harmonic Forces
        f_harm = -2.0 * k * (dist - r0)

        # Lennard-Jones Forces
        safe_dist = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
        sr6 = (sigma / safe_dist) ** 6
        sr12 = sr6 ** 2
        f_lj = (24.0 * epsilon / safe_dist) * (2.0 * sr12 - sr6)
        f_lj = torch.where(safe_dist < lj_cutoff, f_lj, torch.zeros_like(f_lj))
        
        # 3. Apply routing
        force_mag = torch.where(is_harmonic, f_harm, f_lj)
    
    else:
        raise ValueError()

    # Virial contribution: F_y * r_y
    virial_term = force_mag * (dy**2) / (dist + 1e-12)

    # Sum over all edges (0.5 to correct for bidirectional graph)
    virial_sum_y = 0.5 * torch.sum(virial_term)

    # For 2D systems: Total KE = N * kB * T; Y-component is 0.5 * Total KE
    kinetic_pressure_val = (0.5 * num_particles * KB_METAL * temperature) / (volume + 1e-12)

    # Total pressure
    virial_pressure_val = virial_sum_y / (volume + 1e-12)
    p_yy_total = virial_pressure_val + kinetic_pressure_val

    # Inertial update (Langevin piston logic)
    driving_force = (p_yy_total - target_pressure) * lx

    # Apply damping
    total_force = driving_force - (damping * box_vel_y)

    # Acceleration from F = ma
    box_acc = total_force / W_y

    # Integration
    new_box_vel_y = box_vel_y + box_acc * stride_dt

    # L_new = L_old + (v * dt)
    new_ly = ly + (new_box_vel_y * stride_dt)

    return new_ly, new_box_vel_y


def estimate_initial_box_vel_y(prev_graph: Data, curr_graph: Data, stride_dt: float) -> Tensor:
    ly_p = prev_graph.box_tensor[1]
    ly_c = curr_graph.box_tensor[1]

    # v(t) approx (L_c - L_p) / dt
    box_vel_y = (ly_c - ly_p) / stride_dt

    return box_vel_y


def estimate_initial_box_vel_y_accurate(prev_prev_graph: Data, prev_graph: Data, curr_graph: Data, stride_dt: float) -> Tensor:
    ly_pp = prev_prev_graph.box_tensor[1]
    ly_p = prev_graph.box_tensor[1]
    ly_c = curr_graph.box_tensor[1]

    # v(t) approx [3*L_t - 4*L_{t-1} + L_{t-2}] / (2*dt)
    box_vel_y = (3 * ly_c - 4 * ly_p + ly_pp) / (2 * stride_dt)
    
    return box_vel_y