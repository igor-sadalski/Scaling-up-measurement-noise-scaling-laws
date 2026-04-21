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

## Fine-tuning a pretrained SE STATE model

Companion script: `analysis/2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py`
Plotting notebook: `analysis/2026-04-21_14-00_plotting_finetune_pretrained_state_pbmc_noise_scaling.ipynb`

The script runs a 1-epoch fine-tune of the Arc Institute SE-100M checkpoint across the full PBMC noise-scaling grid (10 sizes x 10 qualities) and writes all artifacts under `$NOISE_SCALING_OUTPUT_BASE/finetunning_state/`.

### 1. Download the pretrained checkpoint

The compute script does this automatically on first run via `huggingface_hub.snapshot_download`. Manual equivalent:

```bash
huggingface-cli download arcinstitute/SE-100M \
    --local-dir ~/noise_scaling/data/other/finetunning_state/pretrained/SE-100M
```

The snapshot is expected to contain one `*.ckpt`, a `config.yaml` (with a `model` section defining `emsize/d_hid/nhead/nlayers/output_dim`), and an `all_embeddings*.pt` gene-vocabulary file. The locator in `compute_finetune_pretrained_state_pbmc_noise_scaling.py::locate_pretrained` will report which file it picked for each role.

### 2. Run the sweep

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling
python analysis/2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py
```

The script is resume-friendly: (size, quality) cells whose `result.json` status is `ok` are skipped on re-runs. Tune parallelism via `JOBS_PER_GPU` at the top of the script (start at 1; SE-100M at `batch_size=64` + `pad_length=2048` uses ~10-12 GB per job).

### 3. Mechanics — no patches to the STATE package

- PBMC data is re-preprocessed with SE-100M's own gene embeddings (via `state emb preprocess --all-embeddings`) so `ds_emb_mapping` and `valid_genes_masks` line up with the pretrained `pe_embedding`.
- A weights-only ckpt (`finetune_init.ckpt`) is derived once: strip optimizer states, LR-scheduler states, and `epoch`/`global_step` so Lightning's auto-resume loads weights but starts the new fit at epoch 0 with a fresh OneCycleLR.
- `state emb fit` is invoked with `experiment.num_epochs=1`, `optimizer.max_lr=1e-5`, `optimizer.reset_lr_on_restart=true`. Early stopping (patience=5, monitor=`validation/val_loss`) stays enabled.

### 4. Output layout (mirrors `model_sizing/` so the plotting notebook reuses collection logic)

```
$NOISE_SCALING_OUTPUT_BASE/finetunning_state/
    pretrained/SE-100M/                 # HF snapshot (one-time)
        finetune_init.ckpt              # weights-only ckpt derived from above
    finetune_00/<size>/<quality>/
        config.json, result.json
        state_data/                     # SE-100M-vocab profile
        checkpoints/state_<profile>/
        loss/metrics.csv                # copy of Lightning metrics.csv
        embeddings.csv, embeddings.npy
        train_loss.txt, val_loss.txt, test_loss.txt
        MI/<seed>/Y_<signal>_<quality>/lmi_mutual_information.txt
    sweep_results.csv
    finetune_00.yaml
```

## License

STATE is licensed under CC BY-NC-SA 4.0 (non-commercial). See `state/MODEL_LICENSE.md`.
