from collections import namedtuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graph_utils import get_correct_edge_attr


class DifferentiableCompression64(nn.Module):
    def __init__(
        self,
        num_particles: int,
        mass: float = 1.0e6,
        dt: float = 0.01,
        damp: float = 10.0,
        ptemp: float = 1.0e-4,
        pdamp: float = 1000.0,
        target_p: float = 0.0,
        srate: float = 1.0e-5,
        temp_langevin: float = 1.0e-7,
        inertia_prefactor: float = 1.0,
        factor_two: bool = True,
    ):
        super().__init__()
        self.mass = mass
        self.dt = dt
        self.pdamp = pdamp
        self.ptemp = ptemp
        self.damp = damp
        self.target_p = target_p
        self.srate = srate
        self.num_particles = num_particles

        # Metal Units Constants
        self.kb_metal = 8.6173303e-5
        self.mvv2e = 1.0364269e-4

        if factor_two:
            # dof_factor = (2.0 * num_particles) + 2
            dof_factor = (2.0 * num_particles) - 3
            self.W_y = (
                dof_factor * inertia_prefactor * self.kb_metal * self.ptemp * (pdamp**2)
            )
        else:
            dof_factor = num_particles + 1
            self.W_y = (
                dof_factor * inertia_prefactor * self.kb_metal * self.ptemp * (pdamp**2)
            )

        self.target_p_raw = target_p
        self.temp_langevin = temp_langevin
        self.noise_sigma = torch.sqrt(
            torch.tensor(
                2
                * self.kb_metal
                * self.temp_langevin
                * (self.mass * self.mvv2e)
                / (self.damp * self.dt),
                dtype=torch.float64,
            )
        )

    def get_initial_rest_lengths(self, pos: Tensor, edge_index: Tensor, box_size: Tensor) -> Tensor:
        sender, receiver = edge_index
        r_vec = pos[sender] - pos[receiver]
        box_tensor = box_size.view(1, 2).to(torch.float64)
        r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor
        return torch.norm(r_vec, dim=1)

    def compute_forces_and_virial(
        self,
        pos: Tensor,
        box_size: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        r0: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sender, receiver = edge_index
        r_vec = pos[sender] - pos[receiver]
        box_tensor = box_size.view(1, 2).to(dtype=torch.float64)
        r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor
        dist: Tensor = torch.norm(r_vec, dim=1)

        k = edge_attr[:, -1]

        # force_mag = -4.0 * k * (dist - r0)
        force_mag = -2.0 * k * (dist - r0)

        unit_vec = r_vec / dist.unsqueeze(1)
        force_vec = unit_vec * force_mag.unsqueeze(1)

        forces = torch.zeros_like(pos, dtype=torch.float64)
        forces = forces.index_add(0, sender, force_vec)

        virial_y = 0.5 * torch.sum(force_vec[:, 1] * r_vec[:, 1])
        return forces, virial_y

    def zero_momentum(self, x: Tensor, v: Tensor) -> Tensor:
        # Zero linear momentum (Center of Mass velocity)
        v_com = torch.mean(v, dim=0, keepdim=True)
        v_zero_lin = v - v_com

        # Compute positions relative to the Center of Mass
        x_com = torch.mean(x, dim=0, keepdim=True)
        r = x - x_com

        # Calculate 2D Angular Velocity (omega)
        # Angular momentum L_z = sum(r_x * v_y - r_y * v_x)
        # Moment of inertia I = sum(r_x^2 + r_y^2)
        # (Assuming uniform mass across all particles, mass cancels out in omega = L / I)
        rx, ry = r[:, 0], r[:, 1]
        vx, vy = v_zero_lin[:, 0], v_zero_lin[:, 1]

        L_z = torch.sum(rx * vy - ry * vx)
        I = torch.sum(rx**2 + ry**2)

        # Add epsilon to prevent division by zero in perfectly compact states
        omega = L_z / (I + 1e-12)

        # 4. Calculate and subtract rotational velocity (v_rot = omega x r)
        v_rot_x = -omega * ry
        v_rot_y = omega * rx
        v_rot = torch.stack([v_rot_x, v_rot_y], dim=1)

        v_final = v_zero_lin - v_rot
        return v_final

    @torch.compile()
    def forward(
        self,
        x: Tensor,
        v: Tensor,
        box_dims,
        box_v_y: Tensor,
        eta_v: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        r0: Tensor,
        step_idx: int,
        lx_0,
    ):
        dt = self.dt
        dt_2 = dt * 0.5
        lx, ly = box_dims[0], box_dims[1]

        # 1. Predict forces
        forces, virial_y = self.compute_forces_and_virial(
            x, box_dims, edge_index, edge_attr, r0
        )

        kinetic_y = torch.sum(self.mass * v[:, 1] ** 2) * self.mvv2e
        vol = lx * ly * 0.2
        p_yy_raw = (kinetic_y + virial_y) / vol

        driving_energy = (p_yy_raw - self.target_p_raw) * vol
        acc_eps_y = driving_energy / self.W_y

        # 2. Barostat predictor (Sub-Cycled)
        nc = 5
        dt_sub = dt_2 / nc

        target_ke = 0.5 * self.kb_metal * self.ptemp
        Q_b = self.kb_metal * self.ptemp * (self.pdamp**2)

        box_v_y_half = box_v_y
        mtk_scale = 1.0

        # Unpack the 3-chain
        eta_v1, eta_v2, eta_v3 = eta_v[0], eta_v[1], eta_v[2]

        for _ in range(nc):
            eta_acc3 = (Q_b * eta_v2**2 - target_ke) / Q_b
            eta_v3 = eta_v3 + (dt_sub * 0.5) * eta_acc3

            eta_acc2 = (Q_b * eta_v1**2 - target_ke) / Q_b
            eta_v2 = eta_v2 + (dt_sub * 0.5) * (eta_acc2 - eta_v2 * eta_v3)

            box_ke = 0.5 * self.W_y * (box_v_y_half**2)
            eta_acc1 = (box_ke - target_ke) / Q_b
            eta_v1 = eta_v1 + (dt_sub * 0.5) * (eta_acc1 - eta_v1 * eta_v2)

            # Update box velocity
            box_v_y_half = box_v_y_half + dt_sub * (acc_eps_y - eta_v1 * box_v_y_half)
            mtk_scale = mtk_scale * torch.exp(-dt_sub * box_v_y_half)

            # Update thermostats
            box_ke_new = 0.5 * self.W_y * (box_v_y_half**2)
            eta_acc1_new = (box_ke_new - target_ke) / Q_b
            eta_v1 = eta_v1 + (dt_sub * 0.5) * (eta_acc1_new - eta_v1 * eta_v2)

            eta_acc2_new = (Q_b * eta_v1**2 - target_ke) / Q_b
            eta_v2 = eta_v2 + (dt_sub * 0.5) * (eta_acc2_new - eta_v2 * eta_v3)

            eta_acc3_new = (Q_b * eta_v2**2 - target_ke) / Q_b
            eta_v3 = eta_v3 + (dt_sub * 0.5) * eta_acc3_new

        # 3. Particle predictor
        f_random = torch.randn_like(v) * self.noise_sigma

        # LAMMPS-line Langevin
        friction = -(self.mass * self.mvv2e / self.damp) * v
        f_total = forces + friction + f_random

        v_half_update = dt_2 * (f_total / (self.mass * self.mvv2e))
        v_half_x = v[:, 0] + v_half_update[:, 0]
        v_half_y = (v[:, 1] * mtk_scale) + v_half_update[:, 1]
        v_half = torch.stack([v_half_x, v_half_y], dim=1)

        # 4. Position update
        current_time = (step_idx + 1) * dt
        lx_new = lx_0 * (1.0 - self.srate * current_time)
        ly_new = ly * torch.exp(box_v_y_half * dt)

        x_new_x = x[:, 0] * (lx_new / lx) + v_half[:, 0] * dt
        x_new_y = x[:, 1] * (ly_new / ly) + v_half[:, 1] * dt
        x_new = torch.stack([x_new_x, x_new_y], dim=1)

        new_box_tensor = torch.stack([lx_new, ly_new])

        # 5. Corrector forces
        forces_new, virial_y_new = self.compute_forces_and_virial(
            x_new, new_box_tensor, edge_index, edge_attr, r0
        )

        kinetic_y_new = torch.sum(self.mass * v_half[:, 1] ** 2) * self.mvv2e
        vol_new = lx_new * ly_new * 0.2
        p_yy_new = (kinetic_y_new + virial_y_new) / vol_new
        acc_eps_y_new = ((p_yy_new - self.target_p_raw) * vol_new) / self.W_y

        # 6. Barostat correct
        box_v_y_new = box_v_y_half
        mtk_scale_new = 1.0

        for _ in range(nc):

            eta_acc3_corr = (Q_b * eta_v2**2 - target_ke) / Q_b
            eta_v3 = eta_v3 + (dt_sub * 0.5) * eta_acc3_corr

            eta_acc2_corr = (Q_b * eta_v1**2 - target_ke) / Q_b
            eta_v2 = eta_v2 + (dt_sub * 0.5) * (eta_acc2_corr - eta_v2 * eta_v3)

            box_ke_corr = 0.5 * self.W_y * (box_v_y_new**2)
            eta_acc1_corr = (box_ke_corr - target_ke) / Q_b
            eta_v1 = eta_v1 + (dt_sub * 0.5) * (eta_acc1_corr - eta_v1 * eta_v2)

            box_v_y_new = box_v_y_new + dt_sub * (acc_eps_y_new - eta_v1 * box_v_y_new)
            mtk_scale_new = mtk_scale_new * torch.exp(-dt_sub * box_v_y_new)

            box_ke_corr_new = 0.5 * self.W_y * (box_v_y_new**2)
            eta_acc1_corr_new = (box_ke_corr_new - target_ke) / Q_b
            eta_v1 = eta_v1 + (dt_sub * 0.5) * (eta_acc1_corr_new - eta_v1 * eta_v2)

            eta_acc2_corr_new = (Q_b * eta_v1**2 - target_ke) / Q_b
            eta_v2 = eta_v2 + (dt_sub * 0.5) * (eta_acc2_corr_new - eta_v2 * eta_v3)

            eta_acc3_corr_new = (Q_b * eta_v2**2 - target_ke) / Q_b
            eta_v3 = eta_v3 + (dt_sub * 0.5) * eta_acc3_corr_new

        eta_v_new = torch.stack([eta_v1, eta_v2, eta_v3])

        # 7. Particle corrector
        friction_new = -(self.mass * self.mvv2e / self.damp) * v_half
        f_total_new = forces_new + friction_new + f_random

        v_new_update = dt_2 * (f_total_new / (self.mass * self.mvv2e))
        v_new_x = v_half[:, 0] + v_new_update[:, 0]
        v_new_y = (v_half[:, 1] * mtk_scale_new) + v_new_update[:, 1]
        v_new = torch.stack([v_new_x, v_new_y], dim=1)

        # 8. Zero the momentum every 100 MD steps (like LAMMMPS)
        if step_idx % 100 == 0:
            v_new = self.zero_momentum(x_new, v_new)

        return (
            x_new,
            v_new,
            new_box_tensor,
            box_v_y_new,
            eta_v_new,
            p_yy_new,
            kinetic_y_new,
            virial_y_new,
        )

    def update_graph_simulator(self, old_graph: Data, new_x: Tensor, new_box: Tensor) -> Data:
        dummy = Data(
            x=new_x,
            edge_index=old_graph.edge_index,
            edge_attr=old_graph.edge_attr,
            box_tensor=new_box,
        )
        correct_edge_attr = get_correct_edge_attr(dummy, recompute_stiff=False).double()

        return Data(
            x=dummy.x,
            edge_index=dummy.edge_index,
            edge_attr=correct_edge_attr,
            box_tensor=new_box,
            dtype=torch.float64,
        )

    def run_simulator(
        self,
        initial_data: Data,
        steps: int,
        r0: Tensor,
        debug: bool = False,
        device: str = "cuda",
    ) -> tuple[list[Data], list]:
        x = initial_data.x.double()
        v = torch.zeros_like(x).double()

        if hasattr(initial_data, "box_tensor") and isinstance(initial_data.box_tensor, Tensor):
            box = initial_data.box_tensor.double()
        else:
            raise AttributeError("Input graph does not have a `box_tensor` attribute.")

        lx_0 = box[0].clone()
        box_v_y = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float64)
        eta_v = torch.zeros(3, device=device, dtype=torch.float64, requires_grad=True)

        edge_index = initial_data.edge_index
        edge_attr = initial_data.edge_attr.double()

        # Collect graphs
        trajectory = [initial_data]
        conditions = []
        Step = namedtuple("Step", ["pyy", "vyy", "kin"])

        # Run Sim Loop
        curr_x, curr_v = x, v
        curr_box = box

        for i in range(steps):
            curr_x, curr_v, curr_box, box_v_y, eta_v, pyy, kinetic_y, virial_y = (
                self.forward(
                    curr_x,
                    curr_v,
                    curr_box,
                    box_v_y,
                    eta_v,
                    edge_index,
                    edge_attr,
                    r0,
                    step_idx=i,
                    lx_0=lx_0,
                )
            )
            curr_graph = self.update_graph_simulator(initial_data, curr_x, curr_box)
            trajectory.append(curr_graph)

            s = Step(pyy=pyy, kin=kinetic_y, vyy=virial_y)
            conditions.append(s)

            if debug:
                print(f"Step {i:>4}, Pyy={pyy:.4e}, Kinetic={kinetic_y:.4e}, Virial={virial_y:.4e}")

        return trajectory, conditions

    def run_simulator_to_strain(
        self,
        initial_data: Data,
        target_strain: float,
        r0: Tensor,
        debug: bool = False,
        device: str = "cuda",
    ) -> tuple[list[Data], list]:
        """
        Runs the simulator until a specific engineering strain is reached.
        """
        x = initial_data.x.double()
        x = x + torch.randn_like(x) * 1e-6
        v = torch.zeros_like(x).double()

        if hasattr(initial_data, "box_tensor") and isinstance(initial_data.box_tensor, Tensor):
            box = initial_data.box_tensor.double()
        else:
            raise AttributeError("Input graph does not have a `box_tensor` attribute.")

        lx_0 = box[0].clone()
        box_v_y = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float64)
        eta_v = torch.zeros(3, device=device, dtype=torch.float64, requires_grad=True)

        edge_index = initial_data.edge_index
        edge_attr = initial_data.edge_attr.double()

        # Collect graphs
        trajectory = [initial_data]
        conditions = []
        Step = namedtuple("Step", ["pyy", "vyy", "kin"])

        # Run sim Loop
        curr_x, curr_v = x, v
        curr_box = box

        step_idx = 0
        current_strain = 0.0

        # Loop until target strain
        while current_strain < target_strain:
            curr_x, curr_v, curr_box, box_v_y, eta_v, pyy, kinetic_y, virial_y = (
                self.forward(
                    curr_x,
                    curr_v,
                    curr_box,
                    box_v_y,
                    eta_v,
                    edge_index,
                    edge_attr,
                    r0,
                    step_idx=step_idx,
                    lx_0=lx_0,
                )
            )
            curr_graph = self.update_graph_simulator(initial_data, curr_x, curr_box)
            trajectory.append(curr_graph)

            s = Step(pyy=pyy, kin=kinetic_y, vyy=virial_y)
            conditions.append(s)

            # Calculate achieved strain to check loop condition
            current_time = (step_idx + 1) * self.dt
            current_strain = self.srate * current_time

            if debug:
                print(f"Step {step_idx:>4}, Strain={current_strain:.4e}, Pyy={pyy:.4e}, Kinetic={kinetic_y:.4e}, Virial={virial_y:.4e}")

            step_idx += 1

        if debug and current_strain >= target_strain:
            print(f"Target strain {target_strain:.4e} achieved at step {step_idx}.")

        return trajectory, conditions
