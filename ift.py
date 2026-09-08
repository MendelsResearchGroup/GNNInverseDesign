import torch
import torchopt
from torch import Tensor
from torch_geometric.data import Data

from barostat_utils import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from graph_utils import LJInteractionParams, get_correct_edge_attr
from itpo_weights import ITPOWeights
from torch_simulator_wLJ_64 import DifferentiableCompression64
from training_utils import GNNModel, ModelInputs
from utils import (
    build_velocity_graph_correction,
    compute_combined_physics_loss,
    to_f32,
    to_f64,
)


class ImplicitPhysicsRefinement(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_nn: Tensor,
        input_graph: Data,
        model_inputs: ModelInputs,
        barostat_config: dict,
        box_delta_x: float,
        r0: Tensor,
        lj_params: LJInteractionParams | None,
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
            damping_det = damping.detach() if isinstance(damping, torch.Tensor) else damping

            for _ in range(itpo_weights.refinement_iterations):
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
                    # compressed_lx = predicted_graph.box_tensor[0] * bcf_det
                    compressed_lx = predicted_graph.box_tensor[0] + bcf_det

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
                        lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                        target_pressure=barostat_config["target_pressure"],
                        temperature=barostat_config["temperature"],
                    )

                    new_box_tensor = torch.stack([compressed_lx, compressed_ly])

                    # Update graph with a new box
                    predicted_graph.box_tensor = new_box_tensor

                    function_output = get_correct_edge_attr(
                        predicted_graph,
                        recompute_stiff=False,
                        lj_params=lj_params,
                        panic_at_nontensor_box=True,
                    )
                    if isinstance(function_output, Tensor):
                        predicted_graph.edge_attr = function_output
                    elif isinstance(function_output, tuple):
                        edge_index, edge_attr = function_output
                        predicted_graph.edge_index = edge_index
                        predicted_graph.edge_attr = edge_attr

                    # Calculate Losses
                    loss_anchor = torch.mean((a_refined - a_nn.detach()) ** 2)
                    energy_loss, force_loss, pressure_loss = (
                        compute_combined_physics_loss(
                            predicted_graph,
                            r0=r0_det,
                            lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                            target_Pyy=0.0
                        )
                    )

                    loss_physics = (
                        itpo_weights.lambda_energy * energy_loss
                        + itpo_weights.lambda_force * force_loss
                        + itpo_weights.lambda_pressure * pressure_loss
                    )

                    total_loss = loss_anchor + loss_physics
                    total_loss.backward()

                optimizer.step()

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

        # New: added lj_params for later
        ctx.lj_params = lj_params

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

        lj_params = ctx.lj_params

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
            predicted_graph: Data = Data(
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
                lj_cutoff=lj_params.cutoff if lj_params is not None else None,
                target_pressure=barostat_config["target_pressure"],
                temperature=barostat_config["temperature"],
            )

            new_box_tensor = torch.stack([compressed_lx, compressed_ly])

            # Update graph with a new box
            predicted_graph.box_tensor = new_box_tensor

            function_output = get_correct_edge_attr(
                predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True
            )
            if isinstance(function_output, Tensor):
                predicted_graph.edge_attr = function_output
            elif isinstance(function_output, tuple):
                edge_index, edge_attr = function_output
                predicted_graph.edge_index = edge_index
                predicted_graph.edge_attr = edge_attr

            energy_loss, force_loss, pressure_loss = compute_combined_physics_loss(
                predicted_graph, r0=r0, lj_cutoff=lj_params.cutoff if lj_params is not None else None, target_Pyy=0.0
            )

            loss_anchor = torch.mean((a_refined - a_nn) ** 2)

            loss_physics = (
                itpo_weights.lambda_energy * energy_loss
                + itpo_weights.lambda_force * force_loss
                + itpo_weights.lambda_pressure * pressure_loss
            )

            total_loss = loss_anchor + loss_physics

            # 3. Calculate the first derivative (Gradient) at the minimum
            grad_a = torch.autograd.grad(total_loss, a_refined, create_graph=True)[0]

            # 4. Define the Matrix-Vector (Hessian-Vector) product closure for torchopt
            def hvp_closure(v):
                return torch.autograd.grad(grad_a, a_refined, grad_outputs=v, retain_graph=True)[0]

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

        return implicit_gradient, None, None, None, None, None, None, None, None


def physical_inference_step_implicit(
    model: GNNModel,
    input_graph: Data,
    model_inputs: ModelInputs,
    barostat_config: dict,
    box_delta_x: float,
    r0: Tensor,
    lj_params: LJInteractionParams,
    current_box_vel_y: Tensor,
    itpo_weights: ITPOWeights,
) -> tuple[Data, Tensor]:

    # 1. GNN predicts initial accelerations
    a_nn = model(input_graph)
    a_nn = model.output_normalizer.inverse(a_nn)  # Real space accelerations

    # 2. Refine acceleration via Implicit Differentiation
    # This completely replaces the detached inner loop and the STE trick.
    a_step = ImplicitPhysicsRefinement.apply(
        a_nn,
        input_graph,
        model_inputs,
        barostat_config,
        box_delta_x,
        r0,
        lj_params,
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
        lj_cutoff=lj_params.cutoff if lj_params is not None else None,
        target_pressure=barostat_config["target_pressure"],
        temperature=barostat_config["temperature"],
    )

    new_box_tensor = torch.stack([compressed_lx, compressed_ly])

    # Update graph with a new box
    predicted_graph.box_tensor = new_box_tensor
    function_output = get_correct_edge_attr(predicted_graph, recompute_stiff=False, lj_params=lj_params, panic_at_nontensor_box=True)
    if isinstance(function_output, Tensor):
        predicted_graph.edge_attr = function_output
    elif isinstance(function_output, tuple):
        edge_index, edge_attr = function_output
        predicted_graph.edge_index = edge_index
        predicted_graph.edge_attr = edge_attr

    return predicted_graph, new_vel_y


def specialized_rollout_implicit(
    starting_graph: Data,
    gnn_simulator: GNNModel,
    gnn_history: int,
    barostat_config: dict,
    lj_params: LJInteractionParams | None,
    itpo_weights: ITPOWeights,
    md_steps: int,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:

    r0 = starting_graph.edge_attr[:, -2] # okay to blindly extract because only matters for old format edges
    default_skip = barostat_config["default_skip"]
    dt = barostat_config["dt"]
    stride_dt = default_skip * dt

    # Bootstrap with torch_simulator64
    starting_graph = to_f64(starting_graph).to(device)
    simulator: DifferentiableCompression64 = DifferentiableCompression64(
        starting_graph.num_nodes, temp_langevin=0.0, lj_params=lj_params,
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

        predicted_graph, current_box_vel_y = physical_inference_step_implicit(
            model=gnn_simulator,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0,
            lj_params=lj_params,
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights,
        )

        rollout.append(predicted_graph)

    return rollout


def no_bootstrap_rollout_implicit(
    input_graphs: list[Data],
    gnn_simulator: GNNModel,
    gnn_history: int,
    barostat_config: dict,
    lj_params: LJInteractionParams,
    itpo_weights: ITPOWeights,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:

    r0 = input_graphs[0].edge_attr[:, -2] # okay to blindly extract because only matters for old format edges
    default_skip = barostat_config["default_skip"]
    dt = barostat_config["dt"]
    stride_dt = default_skip * dt

    rollout = [g for g in input_graphs]

    b0 = input_graphs[-2].box_tensor[0]
    b1 = input_graphs[-1].box_tensor[0]
    box_delta_x = b1 - b0

    if gnn_history >= 2:
        current_box_vel_y = estimate_initial_box_vel_y_accurate(
            input_graphs[-3], input_graphs[-2], input_graphs[-1], stride_dt
        )
    else:
        current_box_vel_y = estimate_initial_box_vel_y(input_graphs[-2], input_graphs[-1], stride_dt)

    for _ in range(rollout_steps + 1):
        raw_graphs = rollout[-gnn_history - 1 :]
        raw_graphs = [g.cpu().detach() for g in raw_graphs]
        input_graph = build_velocity_graph_correction(raw_graphs).to(device)
        model_inputs = ModelInputs(rollout[-2].to(device), rollout[-1].to(device), None)

        predicted_graph, current_box_vel_y = physical_inference_step_implicit(
            model=gnn_simulator,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0,
            lj_params=lj_params,
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights,
        )

        rollout.append(predicted_graph.cpu().detach())

    return rollout


def specialized_rollout_cascade_implicit(
    starting_graph: Data,
    gnn_models: list[GNNModel],
    barostat_config: dict,
    box_delta_x: float,
    lj_params: LJInteractionParams | None,
    itpo_weights: ITPOWeights,
    rollout_steps: int,
    device: str = "cuda",
) -> list[Data]:
    """Cascade rollout with ITPO refinement differentiated through the IFT.
    """
    for m in gnn_models:
        m.eval()

    r0 = starting_graph.edge_attr[:, -2]

    rollout = [starting_graph.to(device)]

    # No history exists at the start, so there is nothing to estimate the box
    # velocity from; the barostat starts from rest exactly as in rollout_cascade.
    current_box_vel_y = torch.zeros((), dtype=starting_graph.box_tensor.dtype, device=rollout[0].box_tensor.device)

    for _ in range(rollout_steps):
        # Pick the cascade model with history matching the number of available frames.
        history_len = len(rollout)
        if history_len <= len(gnn_models):
            active_model_idx = history_len - 1
        else:
            active_model_idx = len(gnn_models) - 1
        active_model = gnn_models[active_model_idx]

        input_graph = build_velocity_graph_correction(
            rollout[-len(gnn_models) :],
            panic_at_positions=False,
            total_velocity=False,
        ).to(device)

        prev_graph = rollout[-2] if history_len > 1 else rollout[-1]
        curr_graph = rollout[-1]
        model_inputs = ModelInputs(prev_graph, curr_graph, None)

        predicted_graph, current_box_vel_y = physical_inference_step_implicit(
            model=active_model,
            input_graph=input_graph,
            model_inputs=model_inputs,
            barostat_config=barostat_config,
            box_delta_x=box_delta_x,
            r0=r0,
            lj_params=lj_params,
            current_box_vel_y=current_box_vel_y,
            itpo_weights=itpo_weights,
        )

        rollout.append(predicted_graph)

    return rollout
