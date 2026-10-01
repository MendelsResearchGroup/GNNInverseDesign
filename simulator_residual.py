"""GNN simulator that predicts a correction to the harmonic bond forces.

The total acceleration of one coarse step is a = F_harm(x_t) / m * stride_dt**2 + correction. forward() returns it
in the units of the output normalizer, correction + a_harm / std, so huber_loss and every rollout that applies
output_normalizer.inverse(model(...)) work unchanged, while the network itself only learns the normalized residual.

The harmonic forces need the network's rest lengths, which are not part of the input graph: set model.r0 once per
network, as the rollouts do with r0.
"""

from torch import Tensor
from torch_geometric.data import Data

from pressure import compute_per_particle_forces
from simulator_SA_cpu_test import Model


def harmonic_acceleration(data: Data, r0: Tensor, mass: float, stride_dt: float) -> Tensor:
    """One explicit coarse step of the harmonic bond forces at the current positions of `data`."""
    graph = Data(x=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr, box_tensor=data.box_tensor)
    return compute_per_particle_forces(graph, r0=r0, cutoff=None) / mass * stride_dt**2


class ResidualModel(Model):
    def __init__(self, data: Data, hidden_size: int, n_layers: int, num_mlp: int, mass: float, stride_dt: float, device: str = "cuda"):
        super().__init__(data, hidden_size, n_layers, num_mlp, device)
        self.mass = mass
        self.stride_dt = stride_dt
        self.r0: Tensor | None = None

    def forward(self, data: Data, is_training: bool = True) -> Tensor:
        if self.r0 is None:
            raise ValueError("Set model.r0 to the rest lengths of the network.")
        correction = super().forward(data, is_training)
        a_harm = harmonic_acceleration(data, self.r0, self.mass, self.stride_dt)
        return correction + a_harm / self.output_normalizer._std_with_epsilon()
