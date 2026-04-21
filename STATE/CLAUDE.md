# STATE - Claude Code Instructions

## Overview

This directory contains a local deployment of the **STATE** framework (Arc Institute) for learning self-supervised cell embeddings from gene expression data. STATE uses a Transformer encoder with optimal-transport-inspired losses (Wasserstein, energy, Sinkhorn) for masked autoencoder-style pretraining on single-cell RNA-seq.

The framework has two stages:
1. **SE (State Embedding)**: Self-supervised cell embeddings via masked prediction
2. **TX (State Transition)**: Perturbation effect prediction (not used locally)

Only the SE stage is actively used in this project.

## Environment

Activate the conda environment before running Python commands:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling
```

STATE has its own CLI installed from `state/`:
```bash
state emb preprocess ...
state emb fit ...
state emb transform ...
```

## Directory Layout

| Path | Purpose |
|------|---------|
| `train_state.py` | Main training script: loads data, creates manifests, preprocesses, trains SE model, generates UMAP |
| `brainstorm_training.py` | Hyperparameter sweep comparing 4 configs (lr, epochs, batch size) |
| `se_config.yaml` | Custom config overriding STATE defaults for this project's spatial data |
| `state/` | STATE package source (Arc Institute). Inner `README.md` has full CLI docs |
| `state/src/state/` | Core source code |
| `state/src/state/emb/` | Embedding model: `nn/model.py` (StateEmbeddingModel), `nn/flash_transformer.py`, `nn/loss.py`, `inference.py`, `data/loader.py`, `train/trainer.py` |
| `state/src/state/tx/` | Perturbation model (not used locally): `models/state_transition.py`, `models/base.py` |
| `state/src/state/_cli/` | CLI handlers: `_emb/_fit.py`, `_emb/_preprocess.py`, `_emb/_transform.py` |
| `state/src/state/configs/` | Hydra configs: `state-defaults.yaml` (SE defaults), `config.yaml` (TX base), `model/*.yaml` |
| `data/` | H5AD train/val/test data files |
| `se_profile/` | Preprocessed embeddings and gene mappings (output of `state emb preprocess`) |
| `se_checkpoints/` | Model checkpoints from training |
| `brainstorm/` | Outputs from hyperparameter comparison (4 sub-experiments) |

## Key Architecture (SE Model)

Defined in `state/src/state/emb/nn/model.py` (`StateEmbeddingModel`):
- Input: gene expression "sentences" (gene IDs + counts)
- Gene embeddings: ESM-2 protein embeddings or one-hot (configurable)
- Encoder MLP: token_dim -> d_model (SkipBlock residual MLPs)
- Transformer: Flash attention encoder (`flash_transformer.py`)
- Decoder: projects to output_dim for cell embeddings
- Binary decoder: predicts gene presence (sparse reconstruction)
- Loss: TabularLoss (Wasserstein + energy) from `nn/loss.py`
- Optimizer: AdamW with OneCycleLR scheduler

Default config (`state-defaults.yaml`): emsize=256, d_hid=512, nhead=4, nlayers=3, output_dim=256, dropout=0.02, batch_size=64, max_lr=1e-3

## Configuration

STATE uses Hydra for config management. The local `se_config.yaml` overrides defaults:
- `embeddings.current: spatial` (one-hot, 483 genes for small spatial dataset)
- `dataset.current: spatial` (local h5ad files)
- `experiment.checkpoint.path`: local checkpoint directory
- `wandb.enable: false`

Override config values via CLI: `state emb fit --config se_config.yaml model.nlayers=6`

## Training Workflow

1. Create CSV manifests pointing to h5ad files (done in `train_state.py`)
2. Run `state emb preprocess` to create gene embeddings + mappings in `se_profile/`
3. Run `state emb fit` to train the transformer
4. Run `state emb transform` to generate cell embeddings on test data

## Data Format

- H5AD (AnnData) throughout
- Train/val/test splits in `data/`
- Gene embeddings in `se_profile/all_embeddings_spatial.pt`
- Dataset-embedding index mapping in `se_profile/ds_emb_mapping_spatial.torch`

## License

STATE is licensed under CC BY-NC-SA 4.0 (non-commercial). See `state/MODEL_LICENSE.md`.
