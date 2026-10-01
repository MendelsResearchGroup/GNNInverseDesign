from dataclasses import dataclass
from enum import Enum, auto


class ModelType(Enum):
    GNNModel = auto()
    SimulatorCascade = auto()


class DatasetType(Enum):
    NodeOptimized = auto()
    StiffOptimized = auto()
    Noisy = auto()
    LJNoisy = auto()
    LJNoisyLong = auto()
    StiffAngles = auto()
    HighT = auto()

    def __str__(self):
        mapping = {
            DatasetType.NodeOptimized: "node_optimized",
            DatasetType.StiffOptimized: "stiff_optimized",
            DatasetType.Noisy: "noisy",
            DatasetType.LJNoisy: "lj_noisy",
            DatasetType.LJNoisyLong: "lj_noisy_long",
            DatasetType.StiffAngles: "stiff_with_angles",
            DatasetType.HighT: "high_temperature",
        }
        return mapping[self]


@dataclass
class ITPOWeights:
    refinement_iterations: int = 10
    learning_rate: float = 1e-6
    lambda_force: float = 1e-7
    lambda_energy: float = 1e-7
    lambda_pressure: float = 1e-7
    force_tol: float = 1e-8
    pressure_tol: float = 1e-8

params = {
    (ModelType.GNNModel, DatasetType.StiffAngles): ITPOWeights(
        20,
        1.2907612996037686e-06,
        0.11614716904497102,
        1.8895029029527937e-08,
        8.359865403433733e-08,
        1e-8,
        1e-8,
    ),
    (ModelType.GNNModel, DatasetType.Noisy): ITPOWeights(
        20,
        5.7951649814079445e-05,
        1.9718447726118052e-06,
        0.00032656779006556396,
        1.489615655315828e-08,
        1e-8,
        1e-8,
    ),
    (ModelType.GNNModel, DatasetType.NodeOptimized): ITPOWeights(
        30,
        2.5703244417508413e-05,
        3.747396215420535e-05,
        6.716732763896292e-05,
        0.00019581509246310695,
        1e-8,
        1e-8,
    ),
    (ModelType.GNNModel, DatasetType.StiffOptimized): ITPOWeights(
        20,
        8.271896707246153e-07,
        1.0018347178848147e-08,
        0.010020300800691448,
        6.514187230782921e-06,
        1e-8,
        1e-8,
    ),
    (ModelType.GNNModel, DatasetType.LJNoisy): ITPOWeights(
        refinement_iterations=30,
        learning_rate=4.7879350145586726e-08,
        lambda_force=6.364070794372085e-05,
        lambda_energy=1.0388005646656995e-08,
        lambda_pressure=2.97461694778765e-06,
        force_tol=1e-8,
        pressure_tol=1e-8,
    ),
    (ModelType.GNNModel, DatasetType.LJNoisyLong): ITPOWeights(
        refinement_iterations = 30,
        learning_rate =   4.4036227005146560e-05,
        lambda_force =    4.3922113961895840e-06,
        lambda_energy =   1.1523689311519826e-08,
        lambda_pressure = 0.0004826152072281327,
        force_tol=1e-8,
        pressure_tol=1e-8,
    ),
    # (ModelType.SimulatorCascade, DatasetType.NodeOptimized): ITPOWeights(
    #     50,
    #     2.4347939215508626e-06,
    #     0.2571693753028027,
    #     3.0716147827138016e-07,
    #     0.000396262687255812,
    #     1e-8,
    #     1e-8,
    # ),
    
    # Best Trial: R2 Score: 0.8874426802950278
    #     lr: 5.183287915599958e-08
    #     lambda_force: 9.018302326030932e-08
    #     lambda_energy: 4.9045399766971285e-05
    #     lambda_pressure: 5.069919022905043e-07
    (ModelType.SimulatorCascade, DatasetType.NodeOptimized): ITPOWeights(
        30,
        5.183287915599958e-08,
        9.018302326030932e-08,
        4.9045399766971285e-05,
        5.069919022905043e-07,
        1e-8,
        1e-8,
    ),
    (ModelType.SimulatorCascade, DatasetType.StiffOptimized): ITPOWeights(
        35,
        3.0199055709794887e-05,
        0.004990635231284185,
        6.667337380211805e-05,
        0.00032247351656595965,
        1e-8,
        1e-8,
    ),
    (ModelType.SimulatorCascade, DatasetType.NodeOptimized, "full range"): ITPOWeights(
        13,
        9.307810189079177e-06,
        7.889846905670974e-07,
        1.29333379050348e-07,
        0.00034538814676625466,
        1e-8,
        1e-8,
    ),
    (ModelType.SimulatorCascade, DatasetType.StiffOptimized, "full range"): ITPOWeights(
        20,
        0.00028074846353098794,
        0.0002295002544874123,
        8.098829893012657e-05,
        2.936975082119561e-06,
        1e-8,
        1e-8,
    ),
}

def get_params(model_type: ModelType, data_type: DatasetType, full_range: bool = False):

    key = (
        (model_type, data_type)
        if not full_range
        else (model_type, data_type, "full range")
    )
    if key not in params:
        raise NotImplementedError(
            f"No ITPO weights found for {model_type.name} on {data_type.name}"
        )

    return params[key]
