import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graph_utils import LJInteractionParams, get_correct_edge_attr


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
        lj_params: LJInteractionParams | None = None,
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

        # LJ interactions
        self.lj_params = lj_params

        dof_factor = (2.0 * num_particles) - 3
        self.W_y = (dof_factor * inertia_prefactor * self.kb_metal * self.ptemp * (pdamp**2))

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


    def compute_forces_and_virial(
        self,
        pos: Tensor,
        box_size: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        r0: Tensor,  # Step-0 rest lengths for original harmonic bonds
    ) -> tuple[Tensor, Tensor, Tensor]:
        sender, receiver = edge_index
        r_vec = pos[sender] - pos[receiver]
        box_tensor = box_size.view(1, 2).to(dtype=torch.float64)

        # Minimum Image Convention (Periodic Boundary Conditions)
        r_vec = r_vec - torch.round(r_vec / box_tensor) * box_tensor
        dist: Tensor = torch.norm(r_vec, dim=1)

        force_harmonic = torch.zeros_like(dist, dtype=torch.float64)
        num_edge_features = edge_attr.shape[1]

        if num_edge_features == 4:
            k = edge_attr[:, -1]

            # Harmonic Forces: F = -2k * (r - r0)
            force = -2.0 * k * (dist - r0) # force_mag = -4.0 * k * (dist - r0)
        elif num_edge_features == 7:

            # [is_orig, is_lj, vec_x, vec_y, length_or_dist, stiffness_or_epsilon, r0_or_sigma]
            is_orig = edge_attr[:, 0].bool()

            # Extract parameters
            k = edge_attr[:, 5]
            epsilon = edge_attr[:, 5]
            r0 = edge_attr[:, 6]
            sigma = edge_attr[:, 6]
            cutoff = self.lj_params.cutoff

            # Harmonic Forces: F = -2k * (r - r0)
            # Compute for all edges, filter later. Works better with torch.compile() that way
            force_harmonic = -2.0 * k * (dist - r0)

            # Lennard-Jones Forces: F = (24 * eps / r) * (2 * (sig/r)^12 - (sig/r)^6)
            # Same deal, compute for all, filter later
            safe_dist = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
            sr6 = (sigma / safe_dist) ** 6
            sr12 = sr6**2
            force_lj = (24.0 * epsilon / safe_dist) * (2.0 * sr12 - sr6)
            force_lj = torch.where(dist < cutoff, force_lj, torch.zeros_like(force_lj))
            
            force = torch.where(is_orig, force_harmonic, force_lj)
        else:
            raise ValueError(f"Unexpected edge_attr dimension: {num_edge_features}. Expected 4 (old format, no LJ) or 7 (new format with LJ).")

        # Compute direction and scale by magnitude
        safe_dist_unit = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
        unit_vec = r_vec / safe_dist_unit.unsqueeze(1)
        force_vec = unit_vec * force.unsqueeze(1)

        # Accumulate forces onto particles
        forces = torch.zeros_like(pos, dtype=torch.float64)
        forces = forces.index_add(0, sender, force_vec)

        # Calculate virial stress
        virial_x = 0.5 * torch.sum(force_vec[:, 0] * r_vec[:, 0])
        virial_y = 0.5 * torch.sum(force_vec[:, 1] * r_vec[:, 1])

        return forces, virial_x, virial_y


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

    @torch.compile(dynamic=True)
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
        forces, _virial_x, virial_y = self.compute_forces_and_virial(
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
        forces_new, _virial_x_new, virial_y_new = self.compute_forces_and_virial(
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
        # Evaluate the zeroed momentum
        v_zeroed = self.zero_momentum(x_new, v_new)

        # Create a boolean mask on the GPU
        is_zero_step = (step_idx % 100 == 0)

        # Let the GPU select the correct values without leaving CUDA
        v_new = torch.where(is_zero_step, v_zeroed, v_new)

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
        # Can be edge_attr if old format, can be edge_index, edge_attr if new format
        function_output = get_correct_edge_attr(dummy, recompute_stiff=False, lj_params=self.lj_params, panic_at_nontensor_box=True)

        match function_output:
            case Tensor():
                return Data(
                    x=dummy.x,
                    edge_index=dummy.edge_index,
                    edge_attr=function_output.double(),
                    box_tensor=new_box,
                    dtype=torch.float64,
                )

            case (new_edge_index, new_edge_attr):
                return Data(
                    x=dummy.x,
                    edge_index=new_edge_index,
                    edge_attr=new_edge_attr.double(),
                    box_tensor=new_box,
                    dtype=torch.float64,
                )

            case _:
                raise TypeError(f"Unexpected output type: {type(function_output)}")

    @torch.compile(dynamic=True)
    def minimize(
        self,
        graph: Data,
        r0: Tensor,
        max_iter: int = 1000,
        ftol: float = 1e-6,
        stress_tol: float = 1e-7,
        relax_box: bool = True,
        target_stress: float = 0.0,
        box_lr: float = 1e-2,
    ) -> Data:
        
        x = graph.x.clone()
        v = torch.zeros_like(x)
        box_tensor = graph.box_tensor.clone()
        curr_graph = graph

        # FIRE Hyperparameters
        dt = 0.001
        dt_max = 0.05
        dt_min = 1e-5
        f_inc = 1.1
        f_dec = 0.5
        alpha_start = 0.1
        f_alpha = 0.99
        n_min = 5
        
        alpha = alpha_start
        steps_since_negative = 0

        for step in range(1, max_iter + 1):
            forces, virial_x, virial_y = self.compute_forces_and_virial(
                x, box_tensor, curr_graph.edge_index, curr_graph.edge_attr, r0
            )
            
            max_f = torch.max(torch.abs(forces))
            
            # 1. Box Relaxation (Affine Transformation)
            if relax_box:
                # Calculate current static stress (Assuming thickness = 0.2 like in your MD code)
                vol = box_tensor[0] * box_tensor[1] * 0.2
                stress_xx = virial_x / vol
                stress_yy = virial_y / vol
                
                # Calculate strain step (difference between current and target stress)
                # If stress is positive, material wants to push the box outward.
                strain_x = box_lr * (stress_xx - target_stress)
                strain_y = box_lr * (stress_yy - target_stress)
                
                # Update box dimensions
                new_lx = box_tensor[0] * (1.0 + strain_x)
                new_ly = box_tensor[1] * (1.0 + strain_y)
                new_box = torch.stack([new_lx, new_ly])
                
                # Apply affine scaling to particle coordinates
                x_scaled = x.clone()
                x_scaled[:, 0] = x[:, 0] * (new_lx / box_tensor[0])
                x_scaled[:, 1] = x[:, 1] * (new_ly / box_tensor[1])
                
                x = x_scaled
                box_tensor = new_box

            # Stop condition (Strictly require both force AND stress to be resolved)
            if max_f < ftol:
                if not relax_box:
                    break
                else:
                    # Force the stress error to be less than 1e-5 before allowing the minimizer to exit
                    stress_error_x = torch.abs(stress_xx - target_stress)
                    stress_error_y = torch.abs(stress_yy - target_stress)
                    if stress_error_x < stress_tol and stress_error_y < stress_tol:
                        break

            # 2. FIRE Particle Relaxation
            P = torch.sum(forces * v)
            
            if P > 0:
                v_norm = torch.norm(v)
                f_norm = torch.norm(forces) + 1e-12
                v = (1.0 - alpha) * v + alpha * v_norm * (forces / f_norm)
                
                steps_since_negative += 1
                if steps_since_negative > n_min:
                    dt = min(dt * f_inc, dt_max)
                    alpha = alpha * f_alpha
            else:
                v = torch.zeros_like(v)
                dt = max(dt * f_dec, dt_min)
                alpha = alpha_start
                steps_since_negative = 0
                
            v = v + dt * forces 
            x = x + dt * v
            
            # 3. Update dynamic topology
            curr_graph = self.update_graph_simulator(curr_graph, x, box_tensor)
            
        relaxed_graph = Data(
            x=x, pos=x, 
            edge_index=curr_graph.edge_index, 
            edge_attr=curr_graph.edge_attr,
            box_tensor=box_tensor, 
            dtype=torch.float64
        )
        return relaxed_graph


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
        box_v_y = torch.tensor(0.0, device=device, dtype=torch.float64)
        eta_v = torch.zeros(3, device=device, dtype=torch.float64)

        edge_index = initial_data.edge_index
        edge_attr = initial_data.edge_attr.double()

        # Collect graphs
        trajectory = [initial_data]
        conditions = []
        # Step = namedtuple("Step", ["pyy", "vyy", "kin"])

        # Run Sim Loop
        curr_x, curr_v = x, v
        curr_box = box
        curr_graph = initial_data

        step_tensor = torch.tensor(0, device=device, dtype=torch.int32)
        for i in range(steps):
            step_tensor = torch.tensor(i, device=device, dtype=torch.int32) 
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
                    step_idx=step_tensor,
                    lx_0=lx_0,
                )
            )
            curr_graph = self.update_graph_simulator(curr_graph, curr_x, curr_box)
            
            edge_index = curr_graph.edge_index
            edge_attr = curr_graph.edge_attr.double()
            
            trajectory.append(curr_graph)

            # s = Step(pyy=pyy, kin=kinetic_y, vyy=virial_y)
            # conditions.append(s)

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
        box_v_y = torch.tensor(0.0, device=device, dtype=torch.float64)
        eta_v = torch.zeros(3, device=device, dtype=torch.float64)

        edge_index = initial_data.edge_index
        edge_attr = initial_data.edge_attr.double()

        # Collect graphs
        trajectory = [initial_data]
        conditions = []
        # Step = namedtuple("Step", ["pyy", "vyy", "kin"])

        # Run sim Loop
        curr_x, curr_v = x, v
        curr_box = box
        curr_graph = initial_data

        step_idx = 0
        current_strain = 0.0

        # Loop until target strain
        step_tensor = torch.tensor(0, device=device, dtype=torch.int32)
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
                    step_idx=step_tensor,
                    lx_0=lx_0,
                )
            )
            curr_graph = self.update_graph_simulator(curr_graph, curr_x, curr_box)
            
            edge_index = curr_graph.edge_index
            edge_attr = curr_graph.edge_attr.double()
            
            trajectory.append(curr_graph)

            # s = Step(pyy=pyy, kin=kinetic_y, vyy=virial_y)
            # conditions.append(s)

            # Calculate achieved strain to check loop condition
            current_time = (step_idx + 1) * self.dt
            current_strain = self.srate * current_time

            if debug:
                print(f"Step {step_idx:>4}, Strain={current_strain:.4e}, Pyy={pyy:.4e}, Kinetic={kinetic_y:.4e}, Virial={virial_y:.4e}")

            step_idx += 1
            step_tensor = torch.tensor(step_idx, device=device, dtype=torch.int32)

        if debug and current_strain >= target_strain:
            print(f"Target strain {target_strain:.4e} achieved at step {step_idx}.")

        return trajectory, conditions
