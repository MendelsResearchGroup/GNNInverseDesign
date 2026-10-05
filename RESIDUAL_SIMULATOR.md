# Harmonic-residual GNN simulator: why it generalizes out of range

Notes from 2026-10-01 on the node-optimized dataset. The code is `simulator_residual.py` (model) and `residual_simulator_training.py` (OST and MST training plus evaluation). The results are in `residual_simulator/` and are plotted in the last section of `constraint_projection.ipynb`.

## What the model does

The plain bootstrapped simulator (`simulator_SA_cpu_test.Model`) maps the last 3 non-affine velocities and the bond features to a free 2D acceleration per particle:

    a = sigma * out + mu,    v_{t+1} = v_t + a,    x_{t+1} = x_t + v_{t+1}

The residual model keeps the same network, inputs, loss, data and integrator. It adds one explicit coarse step of the harmonic bond forces at the current positions:

    a = a_harm(x_t) + sigma * out + mu
    a_harm = F_harm(x_t) / (m * mvv2e) * dt_stride^2,    F_harm,i = sum_j -2 k_ij (|r_ij| - r0_ij) r_hat_ij

- **Units:** `m * mvv2e` (1e6 * 1.0364269e-4) is the LAMMPS metal-units mass, as used in `torch_simulator_wLJ_64`.
- **Time step:** `dt_stride = 200 * 0.01`.
- **Rest lengths:** `r0` is the network's set of rest lengths, set through `model.r0`.
- **Interface:** `forward()` returns `out + a_harm / sigma`, so `huber_loss`, `update()` and every rollout work unchanged.
- **What is learned:** the network only learns the correction `a_true - a_harm - mu`.

On its own, `a_harm` is a poor one-step predictor:
- it is about 2x too large, has cosine 0.45 with the true acceleration, and R^2 of about -0.9;
- even with the best-fitting scale (about 0.2) it explains only 30% of the x acceleration and 3% of the y acceleration.

It misses three things:
- **Langevin friction.**
- **The barostat-driven motion in y.**
- **Force changes within the 200 MD substeps.** omega * dt_stride is about 0.5 for the stiffest modes.

As a result, the network has to cancel about 80% of the harmonic term. That is why the residual model's one-step validation loss is worse than the plain model's.

## Results (60 test sims, rollouts from the first 4 ground-truth frames, no refinement)

All models are trained on the 99 training sims with nu >= 0.1. In the table, "out" means test sims with nu < 0.1 (34 of 60), which no model saw in training.

| model | nu R^2 at step 50: all / in / out | out-of-range Pearson r, slope (step 50) | max \|F\| / training envelope |
|---|---|---|---|
| plain, OST | 0.61 / 0.97 / -1.66 | 0.87, 0.34 | 8 to 19 |
| benchmark MST (`new_trained_models/.../MST/checkpoint_epoch_120.pt`) | 0.81 / 1.00 / -0.29 | 0.94, 0.43 | 6 to 7 |
| harmonic step only (no network) | 0.24 / 0.94 / -4.13 | 0.99, 1.76 | 1.7 |
| residual, OST | 0.91 / 0.98 / +0.37 | 0.99, 1.21 | 1.5 |
| residual, MST (epoch 109 of a run stopped at 110) | **0.98 / 0.99 / +0.90** | **0.99, 0.99** | about 1.5 |

At step 150, the residual MST model reaches 0.87 / 0.96 / +0.25 (benchmark MST: 0.73 / 0.98 / -0.66), and its position MSE is about 4x lower.

## Why it generalizes: my best understanding

### 1. Plain models compress the out-of-range response

Plain models rank the nu < 0.1 networks reasonably well (r = 0.87 to 0.94), but their predictions are compressed: slope 0.26 to 0.43, with a bias of +0.1 to +0.2. In the scatter plots their predictions flatten out at about 0 to 0.1, whatever the true nu.

Nu emerges from the collective response of the specific network to compression: how much the non-affine relaxation lets the box contract or expand in y. A plain GNN has to learn that response from bond geometry and velocities. Trained only on nu >= 0.1, it learned a response calibrated to that range and regresses toward it outside it. This is the usual failure of learning a function only from inside its range.

### 2. The harmonic term carries the network-specific mechanics exactly

The harmonic step alone ranks the out-of-range networks almost perfectly: r = 0.986 at step 50 and 0.994 at step 150, with in-range r = 0.999. That is despite its poor one-step accuracy and its wrong scale (slope 1.5 to 1.8, too auxetic).

Which networks are more or less auxetic is a property of their elastic mechanics: the bond topology, stiffnesses, rest lengths and therefore the stiffness matrix. `F_harm` encodes these exactly for any network, including ones outside the training range. In a rollout, the relaxation directions and their relative ease come from the actual network rather than from a learned approximation of it.

### 3. The network only has to learn a recalibration that does not depend on nu

The residual models keep the near-perfect ranking (r = 0.99) and fix the scale: slope 1.21 for OST and 0.99 for MST at step 50.

What the correction has to learn is mostly how one explicit harmonic step differs from 200 damped, barostatted MD substeps:
- the Langevin friction,
- the effective step response with omega * dt of about 0.5,
- the coupling to the box.

These depend on the mass, the time step, the damping and the barostat, which are the same for all networks, and much less on whether a network is auxetic. A correction fitted on nu >= 0.1 networks therefore transfers to nu < 0.1 networks. The plain model instead has to learn both the mechanics and the calibration, and the mechanics is the part that does not transfer.

### 4. Physical restoring forces keep rollouts on the physical manifold

Rollout drift moves particles away from force balance. The harmonic term pushes back in exact proportion, so errors are damped instead of compounding:
- residual models keep forces within about 1.5x the training envelope over 150 steps;
- plain models drift to 7 to 19x;
- position MSE stays 4 to 10x lower.

The rollout therefore stays in states whose inputs look like training data, which keeps the learned correction in its validated regime.

### 5. MST improves the calibration but shows how delicate it is

Multi-step training brought the out-of-range slope from 1.2 to about 1.0 and R^2 from +0.37 to +0.90 at step 50. It went through one unstable phase, though: at epoch 29 (the 3-step curriculum stage), 40 of 60 rollouts diverged. My reading is that the network cancels about 80% of a large explicit term, so small errors in that cancellation change the effective stiffness of the step, and with omega * dt of about 0.5 that can push the coarse step into an unstable regime. The run recovered by epoch 39 and improved steadily afterwards.

## What is not established

- **Hypotheses, not ablations.** Points 3 and 4 are interpretations consistent with the data. The direct test of point 3 would be to check that the learned correction is similar for in-range and out-of-range networks, for example the mean correction-to-harmonic ratio per network.
- **One training run per model and one seed.**
- **The benchmark MST reference was trained in a notebook** on a split that cannot be reconstructed.
- **"Out of range" is out of range in nu only.** The test networks come from the same generator. The inverse-design networks are out of range in structure, with soft bonds and an effective coordination below 4, and have not been tested with this model yet.
- **At long horizons the residual models still overshoot:** slope 1.25 to 1.36 at step 150, i.e. slightly too auxetic.
- **The correction is unbounded.** This is a strong inductive bias, not an architectural constraint. Bounding it (e.g. `c * tanh(.)` scaled to its training range) would make it a hard restriction.

## Reproducing

    uv run python residual_simulator_training.py --model plain --training ost --out-dir residual_simulator
    uv run python residual_simulator_training.py --model residual --training ost --out-dir residual_simulator
    uv run python residual_simulator_training.py --model residual --training mst --out-dir residual_simulator

The benchmark MST, harmonic-only and per-checkpoint MST scores were produced by two helper scripts that are not in the repo yet (`mst_reference/`, `harmonic_only/`, `residual_mst/checkpoint_rollouts.csv`).
