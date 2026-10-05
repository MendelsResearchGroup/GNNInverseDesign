"""Residual simulator trained on several datasets at once, shuffled every epoch.

The training sims of every dataset (100 per dataset, nu >= 0.1, as in residual_simulator_training.py) are pooled, and
every sim keeps the barostat config of its dataset. All other settings are those of residual_simulator_training.py.
After training (and every 10 MST epochs) the test sims of every dataset are rolled out with that dataset's config and
scored separately, with a `dataset` column, so the rows compare directly with residual_simulator/<dataset>/.

    uv run python joint_residual_training.py --training ost --datasets node_optimized stiff_optimized noisy
    uv run python joint_residual_training.py --training mst --datasets node_optimized stiff_optimized noisy --mst-epochs 60
"""

import argparse
import os
from copy import deepcopy

import pandas as pd
import torch
from torch_geometric.data import Data

import constraint_projection_benchmark as cpb
import residual_simulator_training as rst
from simulator_residual import ResidualModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=[d for d in rst.DATASETS if "lj_params" not in rst.DATASETS[d]], required=True)
    parser.add_argument("--training", choices=["ost", "mst"], required=True)
    parser.add_argument("--tag", help="output directory name, by default residual_<training>")
    parser.add_argument("--out-dir", default="residual_simulator/joint")
    parser.add_argument("--n-train-sims", type=int, default=100, help="per dataset")
    parser.add_argument("--n-test-sims", type=int, default=60, help="per dataset")
    parser.add_argument("--mst-epochs", type=int, default=rst.MST_EPOCHS, help="the per-dataset MST runs plateau after epoch ~60")
    args = parser.parse_args()
    rst.MST_EPOCHS = args.mst_epochs

    torch.manual_seed(cpb.SEED)
    train_data, train_configs, val_data, tests = [], [], [], {}
    for name in args.datasets:
        dataset = rst.DATASETS[name]
        config = deepcopy(dataset["barostat"])
        config["default_skip"] = rst.DUMP_PERIOD
        train, val, test = rst.load_dataset(dataset, args.n_train_sims, args.n_test_sims)
        envelope = cpb.force_envelope(train, None)  # before training, which moves frames between devices
        tests[name] = (test, envelope, config)
        train_data += train
        train_configs += [config] * len(train)
        val_data += val
        print(f"{name}: {len(train)} training, {len(val)} validation, {len(test)} test sims", flush=True)

    stride_dt = rst.DUMP_PERIOD * train_configs[0]["dt"]
    assert all(rst.DUMP_PERIOD * c["dt"] == stride_dt for c in train_configs)
    init_graph = Data(x=torch.ones((100, rst.HISTORY * 2)), edge_attr=torch.ones((100, train_data[0][0].edge_attr.shape[1])))
    model = ResidualModel(init_graph, 128, 2, 3, rst.MASS, stride_dt).to(rst.DEVICE)

    model_save_directory = os.path.join(args.out_dir, args.tag or f"residual_{args.training}")
    os.makedirs(model_save_directory, exist_ok=True)
    checkpoints = list(range(50, rst.MAX_SIM_LEN - rst.HISTORY, 50))

    def evaluate():
        rows = []
        for name, (test, envelope, config) in tests.items():
            dataset_rows = rst.evaluate_rollouts(model, test, envelope, checkpoints, config)
            print(f"{name}:", flush=True)
            rst.summarize(dataset_rows)
            rows += [{"dataset": name, **row} for row in dataset_rows]
        return rows

    def score(epoch):
        path = os.path.join(model_save_directory, "checkpoint_rollouts.csv")
        pd.DataFrame(evaluate()).assign(epoch=epoch).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"Scored epoch {epoch}.", flush=True)

    if args.training == "ost":
        rst.train(model, train_data, val_data, model_save_directory, shuffle=True)
    else:
        rst.train_mst(model, train_data, model_save_directory, train_configs, score=score, shuffle=True)

    pd.DataFrame(evaluate()).assign(model=os.path.basename(model_save_directory)).to_csv(os.path.join(model_save_directory, "rollouts.csv"), index=False)


if __name__ == "__main__":
    main()
