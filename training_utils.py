import itertools
import os
from typing import Protocol

import torch
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm


class ModelInputs:
    prev_graph: Data
    cur_graph: Data
    target_graph: Data

    prev_position: torch.Tensor
    cur_position: torch.Tensor
    target_position: torch.Tensor

    def __init__(self, prev_data: Data, cur_data: Data, target_data: Data | None):
        self.prev_graph = prev_data
        self.cur_graph = cur_data
        self.target_graph = target_data if target_data is not None else None

        self.prev_position = prev_data.pos
        self.cur_position = cur_data.pos
        self.target_position = target_data.pos if target_data is not None else None


class GNNModel(Protocol):
    def forward(self, graph: Data, is_training: bool) -> Tensor: ...

    def update(self, inputs: ModelInputs, model_output: Tensor) -> Data: ...


def freeze_normalizer(model: GNNModel):
    model.node_normalizer.frozen = True
    model.edge_normalizer.frozen = True
    model.output_normalizer.frozen = True
    return model


def huber_loss(model: GNNModel, model_output: Tensor, model_inputs: ModelInputs, is_training: bool = True):
    target_velocity = model_inputs.target_position - model_inputs.cur_position
    current_velocity = model_inputs.cur_position - model_inputs.prev_position
    target_acceleration = target_velocity - current_velocity
    target_acc_norm = model.output_normalizer(target_acceleration, accumulate=is_training)
    return torch.nn.functional.huber_loss(model_output, target_acc_norm, delta=1.0)


def load_chunks(paths_dict: dict, n_items: int, max_sim_length: int, desc: str):
    if n_items <= 0:
        return []
        
    data = []
    items_to_load = list(itertools.islice(paths_dict.items(), n_items))
    for _, chunk_path in tqdm(items_to_load, desc=desc):
        chunk = torch.load(chunk_path, weights_only=False)
        
        if max_sim_length is not None:
            chunk = chunk[:max_sim_length]
            
        data.append(chunk)
    return data


def load_data_from_paths(
    data_dir: str,
    data_type: str,
    n_train: int = 50,
    n_finetune: int = 0,
    n_test: int = 50,
    n_val: int = 0,
    max_sim_length: int | None = None
) -> dict:
    
    file_maps = {
        'node_optimized': ("very_high_P_paths_0.01_strain.pt", "high_P_paths_0.01_strain.pt", "rest_paths_0.01_strain.pt"),
        'stiffness_optimized': ("very_high_P_paths.pt", "high_P_paths.pt", "rest_paths.pt")
    }

    if data_type not in file_maps:
        raise ValueError(f"Incorrect data type. Expected 'node_optimized' or 'stiffness_optimized', got '{data_type}'.")

    train_file, ft_file, test_file = file_maps[data_type]

    train_paths = torch.load(os.path.join(data_dir, train_file), weights_only=False)
    ft_paths = torch.load(os.path.join(data_dir, ft_file), weights_only=False)
    test_paths = torch.load(os.path.join(data_dir, test_file), weights_only=False)


    train_data = load_chunks(train_paths, n_train, max_sim_length, desc="Loading training data")
    ft_data = load_chunks(ft_paths, n_finetune, max_sim_length, desc="Loading finetuning data")

    test_data = []
    val_data = []
    test_val_items = list(itertools.islice(test_paths.items(), n_test + n_val))
    for i, (_, chunk_path) in enumerate(tqdm(test_val_items, desc="Loading Test/Validation data")):
        chunk = torch.load(chunk_path, weights_only=False)
        
        if max_sim_length is not None:
            chunk = chunk[:max_sim_length]

        if i < n_test:
            test_data.append(chunk)
        else:
            val_data.append(chunk)

    return {
        'train': train_data,
        'ft': ft_data,
        'val': val_data,
        'test': test_data
    }
