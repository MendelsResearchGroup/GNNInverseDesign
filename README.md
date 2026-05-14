# GNNInverseDesign

[![arXiv](https://img.shields.io/badge/arXiv-2605.XXXXX-B31B1B.svg)](https://arxiv.org/abs/2605.09495)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.20181262)

Original implementation of the paper *"Enabling Structure-Only Initialization and Out-of-Distribution Generalization in GNN-based Molecular Dynamics Simulators"*. 

This repository contains the codebase for the minimal custom MD engine for differentiable compression of disordered elastic networks, GNN-based MD simulator, GNN-based  MD simulator Cascade framework, as well as the Inference-time Physics-based Optimization (ITPO) and differentiable gnn-coupled barostat implementations.

## Installation

### Setup Environment

Clone the repository:

```bash
# Clone the repository
git clone https://github.com/MendelsResearchGroup/GNNInverseDesign.git
cd GNNInserveDesign
```

Setup the environment

```bash
# Option A: Using uv (Recommended)
uv sync

# Option B: Using Conda
conda create --name gnn-inverse-design --file requirements.txt
conda activate gnn-inverse-design

# Option C: Using Pip
pip install -r requirements.txt

```

## Using Pretrained Models & Demos

Examples of pretrained models for the `node_optimized` dataset are provided in the `./trained_models/node_optimized` directory. You can evaluate these models immediately without running the full training loop by using the provided interactive Jupyter notebooks.

## Dataset

Our full dataset as well as a mini version can be found on [Zenodo Repository](https://doi.org/10.5281/zenodo.20181262).

## Citation

```bibtex
@article{shteingolts2026enabling,
  title={Enabling Structure-Only Initialization and Out-of-Distribution Generalization in GNN-based Molecular Dynamics Simulators},
  author={Shteingolts, SA and Salman, Salman N and Mendels, Dan},
  journal={arXiv preprint arXiv:2605.09495},
  year={2026}
}
```
