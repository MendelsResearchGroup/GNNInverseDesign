"""Headless benchmark: force projection on the rollouts of the residual_simulator_training.py models.

For one dataset, every trained model in residual_simulator/<dataset>/ (plain OST, residual OST and MST, the harmonic
step alone and the benchmark MST checkpoints of new_trained_models/<dataset>/MST, where they exist) is rolled out on the test sims without refinement
and with the force projection of constraint_projection_benchmark (|F_i| <= margin * training force envelope) at every
margin. Writes one CSV row per (model, margin, sim, checkpoint step) to residual_simulator/<dataset>/projection.csv;
plotting lives in constraint_projection.ipynb.

    uv run python residual_projection_benchmark.py --dataset stiff_optimized
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import constraint_projection_benchmark as cpb
import residual_simulator_training as rst
from simulator_residual import ResidualModel
from simulator_SA_cpu_test import Model as VelocityModel

MARGINS = [np.nan, 0.5, 1.0, 1.5, 2.0, 3.0]  # nan: no projection

# tag -> (model class, checkpoint; {results} is the dataset's result directory, {dataset} the dataset). Missing checkpoints are skipped.
MODELS = {
    "plain": ("plain", "{results}/plain/checkpoint_epoch_99.pt"),
    "residual": ("residual", "{results}/residual/checkpoint_epoch_99.pt"),
    "residual_mst": ("residual", "{results}/residual_mst/checkpoint_epoch_149.pt"),
    "harmonic_only": ("harmonic", None),
    "mst_reference_20": ("plain", "new_trained_models/{dataset}/MST/checkpoint_epoch_20.pt"),
    "mst_reference_149": ("plain", "new_trained_models/{dataset}/MST/checkpoint_epoch_149.pt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(rst.DATASETS), required=True)
    parser.add_argument("--results-dir", default="residual_simulator")
    parser.add_argument("--n-train-sims", type=int, default=100)
    parser.add_argument("--n-test-sims", type=int, default=60)
    args = parser.parse_args()

    dataset = rst.DATASETS[args.dataset]
    lj_params = dataset.get("lj_params")
    cutoff = lj_params.cutoff if lj_params is not None else None
    config = deepcopy(dataset["barostat"])
    config["default_skip"] = rst.DUMP_PERIOD
    stride_dt = rst.DUMP_PERIOD * config["dt"]
    result_dir = os.path.join(args.results_dir, args.dataset)

    torch.manual_seed(cpb.SEED)
    train_data, _, test_data = rst.load_dataset(dataset, args.n_train_sims, args.n_test_sims)
    envelope = cpb.force_envelope(train_data, cutoff)
    checkpoints = list(range(50, rst.MAX_SIM_LEN - rst.HISTORY, 50))
    init_graph = Data(x=torch.ones((100, rst.HISTORY * 2)), edge_attr=torch.ones((100, train_data[0][0].edge_attr.shape[1])))
    print(f"{args.dataset}: {len(train_data)} training, {len(test_data)} test sims", flush=True)

    out_path = os.path.join(result_dir, "projection.csv")
    rows = []
    for tag, (kind, checkpoint) in MODELS.items():
        if checkpoint is not None:
            checkpoint = checkpoint.format(results=result_dir, dataset=args.dataset)
            if not os.path.exists(checkpoint):
                print(f"{checkpoint} is missing, skipping {tag}.", flush=True)
                continue

        if kind == "residual":
            model = ResidualModel(init_graph, 128, 2, 3, rst.MASS, stride_dt, cutoff).to(rst.DEVICE)
        elif kind == "plain":
            model = VelocityModel(init_graph, 128, 2, 3).to(rst.DEVICE)
        else:
            model = rst.HarmonicOnly(stride_dt, cutoff)
        if checkpoint is not None:
            model.load_checkpoint(checkpoint)

        for margin in MARGINS:
            refine = cpb.no_refine if np.isnan(margin) else cpb.make_refine_force(margin * envelope, cutoff=cutoff)
            print(f"{tag}, margin {margin}", flush=True)
            with torch.no_grad():
                model_rows = rst.evaluate_rollouts(model, test_data, envelope, checkpoints, config, lj_params, refine)
            rst.summarize(model_rows)
            rows += [{"model": tag, "margin": margin, **row} for row in model_rows]
            pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
