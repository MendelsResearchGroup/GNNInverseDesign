from dataclasses import dataclass
from enum import Enum, auto


class ModelType(Enum):
    GNNModel = auto()
    SimulatorCascade = auto()


class DatasetType(Enum):
    NodeOptimized = auto()
    StiffOptimized = auto()

    def __str__(self):
            mapping = {
                DatasetType.NodeOptimized: "node_optimized",
                DatasetType.StiffOptimized: "stiff_optimized",
            }
            return mapping[self]

@dataclass
class ITPOWeights:
    refinement_iterations: int = 10
    learning_rate: float = 1e-6
    lambda_force: float = 1e-7
    lambda_energy: float = 1e-7
    lambda_pressure: float = 1e-7


params = {
    (ModelType.GNNModel, DatasetType.NodeOptimized): ITPOWeights(
        20,
        9.790375237417396e-06,
        0.0005658108853271458,
        1.9600958725034293e-05,
        1.9624485960300745e-05,
    ),
    (ModelType.GNNModel, DatasetType.StiffOptimized): ITPOWeights(
        19,
        1.9105108674549167e-06,
        0.0001583573157398282,
        2.3068090072225652e-05,
        1.6904617560078452e-07,
    ),
    (ModelType.SimulatorCascade, DatasetType.NodeOptimized): ITPOWeights(
        50,
        2.4347939215508626e-06,
        0.2571693753028027,
        3.0716147827138016e-07,
        0.000396262687255812,
    ),
    (ModelType.SimulatorCascade, DatasetType.StiffOptimized): ITPOWeights(
        35,
        3.0199055709794887e-05,
        0.004990635231284185,
        6.667337380211805e-05,
        0.00032247351656595965,
    ),
}


def get_params(model: ModelType, dataset: DatasetType):

    key = (model, dataset)
    if key not in params:
        raise NotImplementedError(f"No ITPO weights found for {model.name} on {dataset.name}")

    return params[key]
