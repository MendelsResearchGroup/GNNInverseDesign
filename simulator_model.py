import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from training_utils import ModelInputs
from utils import build_mlp, get_correct_edge_vec


class AxisSharedNodeEncoder(torch.nn.Module):
    """
    Encodes node features by treating the two spatial dimensions (x, y)
    as separate batch items sharing the same weights.
    """

    def __init__(self, num_history_steps: int, hidden_dim: int, num_mlp: int):
        super().__init__()
        self.axis_mlp = build_mlp(num_history_steps, hidden_dim, hidden_dim, num_mlp=num_mlp, lay_norm=False)

    def forward(self, x: Tensor) -> Tensor:

        x_reshaped = x.view(x.size(0), 2, -1)
        encoded = self.axis_mlp(x_reshaped)
        return encoded.view(x.size(0), -1)


class Encoder(torch.nn.Module):
    def __init__(self, data: Data, hidden_size: int, num_mlp: int):
        super().__init__()

        self.num_history_steps = data.num_features // 2
        self.shared_node_encoder = AxisSharedNodeEncoder(num_history_steps=self.num_history_steps, hidden_dim=hidden_size, num_mlp=num_mlp)
        self.node_projection = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.edge_encoder = build_mlp(
            data.num_edge_features,
            hidden_size,
            hidden_size,
            num_mlp=num_mlp,
            lay_norm=False,
        )

    def forward(self, data: Data) -> Data:
        shared_features = self.shared_node_encoder(data.x)
        x_encoded = self.node_projection(shared_features)

        return Data(
            x=x_encoded,
            edge_index=data.edge_index,
            edge_attr=self.edge_encoder(data.edge_attr),
            box=data.box if hasattr(data, "box") else None,
            box_tensor=data.box_tensor if hasattr(data, "box_tensor") else None,
        )


class CustomMessagePassing(MessagePassing):
    def __init__(self, hidden_size: int, num_mlp: int):
        super(CustomMessagePassing, self).__init__(aggr="add")
        self.node_layer = build_mlp(hidden_size * 4, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=True)
        self.edge_layer = build_mlp(
            hidden_size * 3,
            hidden_size,
            hidden_size * 3,
            num_mlp=num_mlp,
            lay_norm=True,
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        message_block = torch.cat([x_i, x_j, edge_attr], dim=1)
        message_block = self.edge_layer(message_block)
        return message_block

    def update(self, agg: Tensor, x: Tensor) -> Tensor:
        new_nodes = torch.cat([agg, x], dim=1)
        new_nodes = self.node_layer(new_nodes)
        return new_nodes

    def forward(self, data: Data):
        x = self.propagate(edge_index=data.edge_index, x=data.x, edge_attr=data.edge_attr)
        return Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            box=data.box if hasattr(data, "box") else None,
            box_tensor=data.box_tensor if hasattr(data, "box_tensor") else None,
        )


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__()
        self.node_decoder = build_mlp(hidden_size, hidden_size, 2, num_mlp=num_mlp, lay_norm=False)

    def forward(self, data: Data) -> Tensor:
        predicted_velocity_change = self.node_decoder(data.x)
        return predicted_velocity_change


class Normalizer(torch.nn.Module):
    def __init__(
        self,
        size,
        max_accumulations: int = 1_000_000,
        std_epsilon: float = 1e-8,
        name="Normalizer",
        device="cuda",
    ):
        super(Normalizer, self).__init__()
        self.frozen = False
        self.name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float, requires_grad=False, device=device)
        self._acc_count = torch.tensor(0, dtype=torch.float, requires_grad=False, device=device)
        self._num_accumulations = torch.tensor(0, dtype=torch.float, requires_grad=False, device=device)
        self._acc_sum = torch.zeros((1, size), dtype=torch.float, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float, requires_grad=False, device=device)

    def forward(self, data: Tensor, accumulate=True, is_training: bool = True):
        if accumulate and is_training and not self.frozen:
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(data.detach())
        return (data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: Tensor):
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, data):
        count = data.shape[0]
        data_sum = torch.sum(data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(data**2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(
            self._acc_count,
            torch.tensor(1.0, dtype=torch.float, device=self._acc_count.device),
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(
            self._acc_count,
            torch.tensor(1.0, dtype=torch.float, device=self._acc_count.device),
        )
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        dict = {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }

        return dict


class Model(torch.nn.Module):
    def __init__(self, data: Data, hidden_size: int, n_layers: int, num_mlp: int):
        super().__init__()
        self.node_normalizer = Normalizer(size=data.num_features, name="NodeNormalizer")
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer")
        self.output_normalizer = Normalizer(size=2, name="OutputNormalizer")
        self.encoder = Encoder(data, hidden_size, num_mlp=num_mlp)
        self.gnn_layers = ModuleList()
        self.decoder = Decoder(hidden_size, num_mlp=num_mlp)
        for _ in range(n_layers):
            gnn_layer = CustomMessagePassing(hidden_size, num_mlp=num_mlp)
            self.gnn_layers.append(gnn_layer)
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        norm_nodes = self.node_normalizer(graph.x, is_training=is_training)
        norm_edges = self.edge_normalizer(graph.edge_attr, is_training=is_training)
        return Data(
            x=norm_nodes,
            edge_index=graph.edge_index,
            edge_attr=norm_edges,
            box=graph.box if hasattr(graph, "box") else None,
            box_tensor=graph.box_tensor if hasattr(graph, "box_tensor") else None,
        )

    def forward(self, data: Data, is_training: bool = True) -> torch.Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        data = self.encoder(data)

        for gnn_layer in self.gnn_layers:
            residual = data.x
            data = gnn_layer(data)
            data.x = data.x + residual

        predicted_velocity_change = self.decoder(data)
        return predicted_velocity_change

    @classmethod
    def _recalc_edges(cls, data: Data) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, panic_at_nontensor_box=True)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1]
        new_edge_attr = torch.column_stack([edge_vectors, distances, bond_coeffs])
        return new_edge_attr

    def update(self, inputs: ModelInputs, model_output: Tensor, recalc_edges: bool = False) -> Data:
        predicted_acceleration = model_output
        predicted_acceleration = self.output_normalizer.inverse(predicted_acceleration)
        cur_velocity = inputs.cur_position - inputs.prev_position
        updated_velocity = cur_velocity + predicted_acceleration
        predicted_position = inputs.cur_position + updated_velocity

        return Data(
            x=predicted_position,
            pos=predicted_position,
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=inputs.cur_graph.edge_attr,
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            box_tensor=inputs.cur_graph.box_tensor if hasattr(inputs.cur_graph, "box_tensor") else None,
        )

    def velocity_update(self, inputs: ModelInputs, model_output: Tensor) -> Data:
        predicted_displacement = model_output
        predicted_displacement = self.output_normalizer.inverse(predicted_displacement)
        predicted_position = inputs.cur_position + predicted_displacement

        tmp = Data(
            x=predicted_position,
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=inputs.cur_graph.edge_attr,
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            box_tensor=inputs.cur_graph.box_tensor if hasattr(inputs.cur_graph, "box_tensor") else None,
        )

        new_edge_attr = self._recalc_edges(tmp)

        return Data(
            x=predicted_position,
            pos=predicted_position,
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=new_edge_attr,
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            box_tensor=inputs.cur_graph.box_tensor if hasattr(inputs.cur_graph, "box_tensor") else None,
        )

    def loss(self, model_output: Tensor, inputs: ModelInputs, is_training: bool = True) -> Tensor:
        predicted_velocity_change = model_output
        cur_velocity = inputs.cur_position - inputs.prev_position
        target_velocity = inputs.target_position - inputs.cur_position
        target_velocity_change = target_velocity - cur_velocity
        target_velocity_change_normalized = self.output_normalizer(target_velocity_change, is_training=is_training, accumulate=is_training)
        acceleration_loss = torch.nn.functional.mse_loss(predicted_velocity_change, target_velocity_change_normalized)
        return acceleration_loss

    def velocity_loss(self, model_output: Tensor, inputs: ModelInputs, is_training: bool = True) -> Tensor:
        predicted_displacement = model_output
        target_displacement = inputs.target_position - inputs.cur_position
        target_displacement_normalized = self.output_normalizer(target_displacement, is_training=is_training, accumulate=is_training)
        velocity_loss = torch.nn.functional.mse_loss(predicted_displacement, target_displacement_normalized)
        return velocity_loss

    def save_checkpoint(self, savedir: str):
        model = self.state_dict()
        _output_normalizer = self.output_normalizer.get_variable()
        _node_normalizer = self.node_normalizer.get_variable()
        _edge_normalizer = self.edge_normalizer.get_variable()

        to_save = {
            "model": model,
            "output_normalizer": _output_normalizer,
            "node_normalizer": _node_normalizer,
            "edge_normalizer": _edge_normalizer,
        }

        torch.save(to_save, savedir)

    def load_checkpoint(self, ckpdir: str):
        dicts = torch.load(ckpdir, weights_only=False)
        self.load_state_dict(dicts["model"])

        keys = list(dicts.keys())
        keys.remove("model")

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval("self." + k)
                setattr(object, para, value)
