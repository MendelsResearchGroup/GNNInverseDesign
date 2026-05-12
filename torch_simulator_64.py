from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from utils import get_correct_edge_attr


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
        self.damp = damp
        self.target_p = target_p
        self.srate = srate
        self.num_particles = num_particles

        # Metal Units Constants
        self.kb_metal = 8.6173303e-5
        self.mvv2e = 1.0364269e-4

        if factor_two:
            dof_factor = (2.0 * num_particles) + 2
            self.W_y = dof_factor * inertia_prefactor * self.kb_metal * ptemp * (pdamp**2)
        else:
            dof_factor = num_particles + 1
            self.W_y = dof_factor * inertia_prefactor * self.kb_metal * ptemp * (pdamp**2)

        self.target_p_raw = target_p
        self.temp_langevin = temp_langevin

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
    ) -> Tuple[Tensor, Tensor]:
        sender, receiver = edge_index
        r_vec = pos[sender] - pos[receiver]
        box_tensor = box_size.view(1, 2).to(dtype=torch.float64)
        r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor
        dist: Tensor = torch.norm(r_vec, dim=1)

        k = edge_attr[:, -1]

        force_mag = -2.0 * k * (dist - r0)

        unit_vec = r_vec / dist.unsqueeze(1)
        force_vec = unit_vec * force_mag.unsqueeze(1)

        forces = torch.zeros_like(pos, dtype=torch.float64)
        forces = forces.index_add(0, sender, force_vec)

        virial_y = 0.5 * torch.sum(force_vec[:, 1] * r_vec[:, 1])
        return forces, virial_y

    def zero_momentum(self, v: Tensor) -> Tensor:
        v_com = torch.mean(v, dim=0, keepdim=True)
        return v - v_com

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        box_dims,
        box_v_y: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        r0: Tensor,
        step_idx: int,
        lx_0,
    ):
        dt = self.dt
        dt_2 = dt * 0.5
        lx, ly = box_dims[0], box_dims[1]

        # Deform along X axis (constant engineering strain)
        current_time = (step_idx + 1) * dt
        lx_new = lx_0 * (1.0 - self.srate * current_time)
        x_deformed = torch.stack([x[:, 0] * (lx_new / lx), x[:, 1]], dim=1)


        # Compute forces and virial
        current_box_tensor = torch.stack([lx_new, ly])
        forces, virial_y = self.compute_forces_and_virial(x_deformed, current_box_tensor, edge_index, edge_attr, r0)

        v = self.zero_momentum(v)
        kinetic_y = torch.sum(self.mass * v[:, 1] ** 2) * self.mvv2e
        vol = lx_new * ly
        p_yy_raw = (kinetic_y + virial_y) / vol

        driving_energy = (p_yy_raw - self.target_p_raw) * vol
        acc_eps_y = driving_energy / self.W_y

        # Update box velocity
        box_v_y_half = box_v_y + dt_2 * acc_eps_y

        # Update bead velocity
        noise_sigma = torch.sqrt(
            torch.tensor(
                2 * self.kb_metal * self.temp_langevin * (self.mass * self.mvv2e) / (self.damp * self.dt),
                dtype=torch.float64,
                device=x.device,
            )
        )
        f_random = torch.randn_like(v) * noise_sigma

        mtk_scale = torch.exp(-dt_2 * box_v_y_half)

        friction = -(self.mass * self.mvv2e / self.damp) * v
        f_total = forces + friction + f_random

        v_half_update = dt_2 * (f_total / (self.mass * self.mvv2e))
        v_half_x = v[:, 0] + v_half_update[:, 0]
        v_half_y = (v[:, 1] * mtk_scale) + v_half_update[:, 1]
        v_half = torch.stack([v_half_x, v_half_y], dim=1)

        # Position Update
        ly_new = ly * torch.exp(box_v_y_half * dt)
        x_new_x = x_deformed[:, 0] + v_half[:, 0] * dt
        x_new_y = x_deformed[:, 1] * (ly_new / ly) + v_half[:, 1] * dt
        x_new = torch.stack([x_new_x, x_new_y], dim=1)

        # Corrector step
        new_box_tensor = torch.stack([lx_new, ly_new])
        forces_new, virial_y_new = self.compute_forces_and_virial(x_new, new_box_tensor, edge_index, edge_attr, r0)

        v_half = self.zero_momentum(v_half)

        kinetic_y_new = torch.sum(self.mass * v_half[:, 1] ** 2) * self.mvv2e
        vol_new = lx_new * ly_new
        p_yy_new = (kinetic_y_new + virial_y_new) / vol_new

        acc_eps_y_new = ((p_yy_new - self.target_p_raw) * vol_new) / self.W_y

        box_v_y_new = box_v_y_half + dt_2 * acc_eps_y_new
        mtk_scale_new = torch.exp(-dt_2 * box_v_y_new)

        friction_new = -(self.mass * self.mvv2e / self.damp) * v_half
        f_total_new = forces_new + friction_new + f_random

        v_new_update = dt_2 * (f_total_new / (self.mass * self.mvv2e))
        v_new_x = v_half[:, 0] + v_new_update[:, 0]
        v_new_y = (v_half[:, 1] * mtk_scale_new) + v_new_update[:, 1]
        v_new = torch.stack([v_new_x, v_new_y], dim=1)

        v_new = self.zero_momentum(v_new)

        return (
            x_new,
            v_new,
            torch.stack([lx_new, ly_new]),
            box_v_y_new,
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
        debug: bool = False,
        device: str = "cuda",
        r0: Tensor | None = None,
    ) -> Tuple[list[Data], list]:
        x = initial_data.x.double()
        v = torch.zeros_like(x).double()

        if hasattr(initial_data, "box_tensor") and isinstance(initial_data.box_tensor, Tensor):
            box = initial_data.box_tensor.double()
        else:
            raise AttributeError("Graph does not have a tensor with box info.")

        lx_0 = box[0].clone()
        box_v_y = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float64)

        edge_index = initial_data.edge_index
        edge_attr = initial_data.edge_attr.double()

        # Calculate r0 directly (suitable only for global node optimized networks)
        if r0 is None:
            if debug:
                print("r0 is not provided, computing as 1 / stiffness.")
            r0 = 1.0 / edge_attr[:, -1]
        else:
            if debug:
                print("Using provided r0.")
            pass

        # Collect graphs
        trajectory = [initial_data]
        conditions = []
        Step = namedtuple("Step", ["pyy", "vyy", "kin"])

        # Run Sim Loop
        curr_x, curr_v = x, v
        curr_box = box

        for i in range(steps):
            curr_x, curr_v, curr_box, box_v_y, pyy, kinetic_y, virial_y = self.forward(
                curr_x,
                curr_v,
                curr_box,
                box_v_y,
                edge_index,
                edge_attr,
                r0,
                step_idx=i,
                lx_0=lx_0,
            )
            curr_graph = self.update_graph_simulator(initial_data, curr_x, curr_box)
            trajectory.append(curr_graph)

            s = Step(pyy=pyy, kin=kinetic_y, vyy=virial_y)
            conditions.append(s)

            if debug:
                print(f"Step {i:>4}, Pyy={pyy:.4e}, Kinetic={kinetic_y:.4e}, Virial={virial_y:.4e}")

        return trajectory, conditions
