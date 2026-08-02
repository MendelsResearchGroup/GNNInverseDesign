import torch
import torchopt
from torch import Tensor
from torch_geometric.data import Data

from barostat_utils import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from graph_utils import get_correct_edge_attr
from itpo_weights import ITPOWeights
from torch_simulator_wLJ_64 import DifferentiableCompression64
from training_utils import GNNModel, ModelInputs
from utils import (
    build_velocity_graph_correction,
    compute_combined_physics_loss,
    to_f32,
    to_f64,
)


class ImplicitPhysicsRefinementThreshold(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_nn: Tensor,
        input_graph: Data,
        model_inputs: ModelInputs,
        barostat_config: dict,
        box_delta_x: float,
        r0: Tensor,
        current_box_vel_y: Tensor,
        itpo_weights: ITPOWeights,
    ):

        # 1. Get current and previous graph
        curr_graph = model_inputs.cur_graph
        prev_graph = model_inputs.prev_graph

        # 2. Get all barostat-related parameters
        num_particles = input_graph.num_nodes
        dt = barostat_config["dt"]
        default_skip = barostat_config["default_skip"]
        stride_dt = default_skip * dt
        W_y = barostat_config["C_coupling"] * num_particles * (stride_dt**2)
        damping = barostat_config["damping"] * num_particles * stride_dt

        # 3. Forward pass (No gradients tracked to save memory)
        with torch.no_grad():
            a_refined = a_nn.detach().clone()
            a_refined.requires_grad_(True)
            optimizer = torch.optim.Adam([a_refined], lr=itpo_weights.learning_rate)

            # Detach all historical states
            cx_det = curr_graph.x.detach()
            px_det = prev_graph.x.detach()
            ea_det = curr_graph.edge_attr.detach()
            cb_det = curr_graph.box_tensor.detach()
            r0_det = r0.detach()
            vel_y_det = (
                current_box_vel_y.detach()
                if isinstance(current_box_vel_y, torch.Tensor)
                else current_box_vel_y
            )

            bcf_det = (
                box_delta_x.detach()
                if isinstance(box_delta_x, torch.Tensor)
                else box_delta_x
            )

            # Defensively detach W_y and damping if they somehow inherited gradients
            W_y_det = W_y.detach() if isinstance(W_y, torch.Tensor) else W_y
            damping_det = (
                damping.detach() if isinstance(damping, torch.Tensor) else damping
            )

            # Safely extract tolerances (default to 1e-6 if not yet defined in your class)
            force_tol = getattr(itpo_weights, "force_tol", 1e-8)
            pressure_tol = getattr(itpo_weights, "pressure_tol", 1e-8)

            # Using the previous fixed steps as our absolute ceiling to prevent infinite loops
            max_iters = itpo_weights.refinement_iterations

            for step in range(max_iters):
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
                    compressed_lx = predicted_graph.box_tensor[0] * bcf_det

                    # Get new Ly from barostat
                    compressed_ly, _temp_vel_y = update_box_y_thermodynamic(
                        positions=predicted_graph.pos,
                        edge_index=curr_graph.edge_index,
                        edge_attr=ea_det,
                        current_box=cb_det,
                        r0=r0_det.float().to(predicted_graph.pos.device),
                        box_vel_y=vel_y_det,
                        W_y=W_y_det,
                        damping=damping_det,
                        stride_dt=stride_dt,
                        target_pressure=barostat_config["target_pressure"],
                        temperature=barostat_config["temperature"],
                    )

                    new_box_tensor = torch.stack([compressed_lx, compressed_ly])

                    # Update graph with a new box
                    predicted_graph.box_tensor = new_box_tensor
                    predicted_graph.edge_attr = get_correct_edge_attr(
                        predicted_graph,
                        recompute_stiff=False,
                        panic_at_nontensor_box=True,
                    )

                    # Calculate Losses
                    loss_anchor = torch.mean((a_refined - a_nn.detach()) ** 2)
                    _energy_val, force_loss, pressure_loss = (
                        compute_combined_physics_loss(
                            predicted_graph, r0=r0_det, target_Pyy=0.0
                        )
                    )

                    # EXCLUDED ENERGY LOSS: We rely entirely on internal forces and boundary pressures
                    loss_physics = (
                        itpo_weights.lambda_force * force_loss
                        + itpo_weights.lambda_pressure * pressure_loss
                    )

                    # If you absolutely must track energy for debugging, you can log it,
                    # but it is intentionally omitted from loss_physics.

                    total_loss = loss_anchor + loss_physics
                    total_loss.backward()
                    print(
                        f"Step: {step}: Force loss = {force_loss.item():3e}, pressure loss = {pressure_loss.item():.3e}."
                    )

                optimizer.step()

                # EARLY STOPPING CHECK
                # We use .item() to pull the scalar out of the computational graph.
                # Only evaluate early stopping if the anchor has settled down a bit too,
                # though strictly speaking physics constraints are priority.
                if (force_loss.item() <= force_tol) and (
                    pressure_loss.item() <= pressure_tol
                ):
                    break

        # Save Tensors natively
        ctx.save_for_backward(a_nn, a_refined, r0, current_box_vel_y)

        # Save non-tensors and complex objects as ctx attributes
        ctx.model_inputs = model_inputs
        ctx.barostat_config = barostat_config
        ctx.box_delta_x = box_delta_x
        ctx.itpo_weights = itpo_weights
        ctx.W_y = W_y
        ctx.damping = damping
        ctx.stride_dt = stride_dt

        return a_refined.detach()

    @staticmethod
    def backward(ctx, grad_output):

        a_nn, a_refined, r0, current_box_vel_y = ctx.saved_tensors

        model_inputs = ctx.model_inputs
        barostat_config = ctx.barostat_config
        box_delta_x = ctx.box_delta_x
        itpo_weights = ctx.itpo_weights
        W_y = ctx.W_y
        damping = ctx.damping
        stride_dt = ctx.stride_dt

        curr_graph = model_inputs.cur_graph
        prev_graph = model_inputs.prev_graph

        # 2. Re-enable gradients to build the computation graph ONLY at the minimum
        with torch.enable_grad():
            a_refined.requires_grad_(True)

            a_step = a_refined

            v_curr = curr_graph.x - prev_graph.x
            v_next = v_curr + a_step
            x_next = curr_graph.x + v_next

            # Construct final Data object (Using original inputs to keep grad flowing)
            predicted_graph = Data(
                x=x_next,
                pos=x_next,
                edge_index=curr_graph.edge_index,
                edge_attr=curr_graph.edge_attr,
                box_tensor=curr_graph.box_tensor,
            )

            # Apply uniaxial compression
            compressed_lx = predicted_graph.box_tensor[0] + box_delta_x

            # Get new Ly from barostat
            compressed_ly, _new_vel_y = update_box_y_thermodynamic(
                positions=predicted_graph.pos,
                edge_index=curr_graph.edge_index,
                edge_attr=curr_graph.edge_attr,
                current_box=curr_graph.box_tensor,
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
            predicted_graph.edge_attr = get_correct_edge_attr(
                predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True
            )

            _, force_loss, pressure_loss = compute_combined_physics_loss(
                predicted_graph, r0, target_Pyy=0.0
            )

            loss_anchor = torch.mean((a_refined - a_nn) ** 2)

            loss_physics = (
                +itpo_weights.lambda_force * force_loss
                + itpo_weights.lambda_pressure * pressure_loss
            )

            total_loss = loss_anchor + loss_physics

            # 3. Calculate the first derivative (Gradient) at the minimum
            grad_a = torch.autograd.grad(total_loss, a_refined, create_graph=True)[0]

            # 4. Define the Matrix-Vector (Hessian-Vector) product closure for torchopt
            def hvp_closure(v):
                return torch.autograd.grad(
                    grad_a, a_refined, grad_outputs=v, retain_graph=True
                )[0]

            # 5. Solve the linear system H * x = grad_output using torchopt's CG solver
            # Step A: Instantiate the CG solver with your configuration
            cg_solver = torchopt.linear_solve.solve_cg(maxiter=10)

            # Step B: Call the solver with your Hessian-vector product closure and target vector
            inverse_hvp = cg_solver(matvec=hvp_closure, b=grad_output)

            # 6. Apply the Implicit Function Theorem (IFT)
            # Find N (total number of elements in the acceleration tensor)
            N = a_refined.numel()

            # The mixed derivative is -(2/N) * I
            # IFT is -inverse_hvp * mixed_derivative
            # The negatives cancel out:
            implicit_gradient = inverse_hvp * (2.0 / N)

        return implicit_gradient, None, None, None, None, None, None, None


def physical_inference_step_implicit_threshold(
    model: GNNModel,
    input_graph: Data,
    model_inputs: ModelInputs,
    barostat_config: dict,
    box_delta_x: float,
    r0: Tensor,
    current_box_vel_y: Tensor,
    itpo_weights: ITPOWeights,
) -> tuple[Data, Tensor]:
    raise NotImplementedError
    # 1. GNN predicts initial accelerations
    a_nn = model(input_graph)
    a_nn = model.output_normalizer.inverse(a_nn)  # Real space accelerations

    # 2. Refine acceleration via Implicit Differentiation
    # This completely replaces the detached inner loop and the STE trick.
    a_step = ImplicitPhysicsRefinementThreshold.apply(
        a_nn,
        input_graph,
        model_inputs,
        barostat_config,
        box_delta_x,
        r0,
        current_box_vel_y,
        itpo_weights,
    )

    # 3. Standard Kinematics (PyTorch tracks all of this naturally)
    curr_graph = model_inputs.cur_graph
    prev_graph = model_inputs.prev_graph

    v_curr = curr_graph.x - prev_graph.x
    v_next = v_curr + a_step
    x_next = curr_graph.x + v_next

    # Construct final Data object
    predicted_graph = Data(
        x=x_next,
        pos=x_next,
        edge_index=curr_graph.edge_index,
        edge_attr=curr_graph.edge_attr,
        box_tensor=curr_graph.box_tensor if hasattr(curr_graph, "box_tensor") else None,
    )

    # 4. Barostat Updates
    num_particles = input_graph.num_nodes
    dt = barostat_config["dt"]
    default_skip = barostat_config["default_skip"]
    stride_dt = default_skip * dt
    W_y = barostat_config["C_coupling"] * num_particles * (stride_dt**2)
    damping = barostat_config["damping"] * num_particles * stride_dt

    # Apply uniaxial compression
    compressed_lx = predicted_graph.box_tensor[0] + box_delta_x

    # Get new Ly from barostat
    compressed_ly, new_vel_y = update_box_y_thermodynamic(
        positions=predicted_graph.pos,
        edge_index=curr_graph.edge_index,
        edge_attr=curr_graph.edge_attr,
        current_box=curr_graph.box_tensor,  # Use CURRENT box to calc pressure
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
    predicted_graph.edge_attr = get_correct_edge_attr(
        predicted_graph, recompute_stiff=False, panic_at_nontensor_box=True
    )

    return predicted_graph, new_vel_y


def specialized_rollout_implicit_threshold(
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
        input_graph = build_velocity_graph_correction(rollout[-gnn_history - 1 :]).to(
            device
        )
        model_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)

        predicted_graph, current_box_vel_y = physical_inference_step_implicit_threshold(
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
