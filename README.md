# Scaling Up Measurement Noise Scaling Laws

This repository accompanies our paper studying how single-cell embedding algorithms scale with **dataset size** and **measurement noise** (data quality). We systematically train five embedding methods across four single-cell datasets at 10 logarithmically-spaced sizes and 10 noise levels, then estimate mutual information (MI) between learned representations and biologically meaningful signal variables, and fit power-law scaling curves.

**Scale of the experiment:** 4 datasets x 10 sizes x 10 qualities x 5 algorithms = 2,000 training runs; multiplied by 2--4 signal columns and 4 MI seeds, ~40,000 MI estimates.

---

## Table of contents

- [Public data](#public-data)
- [Repository layout](#repository-layout)
- [Datasets and algorithms](#datasets-and-algorithms)
- [Installation](#installation)
- [Running the experiments](#running-the-experiments)
- [Reproducing figures from the paper](#reproducing-figures-from-the-paper)
- [Pipeline details](#pipeline-details)
- [Mutual information estimation](#mutual-information-estimation)

---

## Public data

All experimental outputs (trained models, embeddings, MI estimates) are publicly available on S3, **no AWS credentials required**:

```
s3://measurement-noise-scaling-laws/data/
```

Programmatic access via `S3Retriever`:

```python
from scaling_laws.s3_retriever import S3Retriever

data = S3Retriever()  # defaults to the public bucket
emb  = data.load_embeddings("merfish", num_cells=7113, quality=1.0,
                            algorithm="Geneformer")
mi   = data.load_mutual_information("merfish", num_cells=7113, quality=1.0,
                                    algorithm="Geneformer", signal="ng_idx", seed=42)
```

---

## Repository layout

```
.
├── scaling_laws/                       # Core package (pip install -e .)
│   └── src/scaling_laws/
│       ├── prepare/
│       │   ├── data.py                 # Experiments + PrepareData orchestrator
│       │   └── utils.py                # Numba-optimized downsampling, t-digest medians
│       ├── algo/
│       │   ├── abc.py                  # BaseAlgorithm interface (with MI estimation)
│       │   ├── geneformer.py           # Geneformer (masked LM, 256-dim)
│       │   ├── scvi.py                 # scVI (VAE, 16-dim)
│       │   ├── state.py                # STATE (self-supervised, 256-dim, subprocess)
│       │   ├── pca.py                  # PCA on HVGs (256-dim)
│       │   ├── rp.py                   # Gaussian random projection (256-dim)
│       │   └── utils.py                # HVG selection, normalization
│       ├── h5ad_reader.py              # Memory-efficient chunked H5AD reader
│       ├── paths.py                    # Single source of truth for paths (env-var overridable)
│       └── s3_retriever.py             # S3 / local data + result loader
│
├── Geneformer/                         # Vendored Geneformer (pip install -e .)
├── STATE/state/                        # Vendored Arc Institute STATE (pip install -e . in `state` env)
│
├── run_pbmc_whole.py                   # Full PBMC sweep (Geneformer, scVI, PCA, RP)
├── run_larry_whole.py                  # Full larry sweep
├── run_merfish_whole.py                # Full merfish sweep
├── run_shendure_whole.py               # Full shendure sweep
├── run_state_shendure_whole.py         # STATE-only on shendure (10 sizes x 10 qualities)
├── run_*_whole.slurm                   # SLURM batch wrappers
├── single_job.py                       # CLI: one (dataset, size, quality, algo) job
├── prepare_hvg.py                      # Pre-compute highly variable gene masks
├── hvg.ipynb                           # Interactive HVG computation
├── requirements.txt                    # Python dependencies for the `modeling` env
│
├── analysis/                           # All post-hoc analyses
│   ├── final_results/                  # Aggregated CSVs and figure PNGs (paper artifacts)
│   ├── 2025-11-18_*                    # Power-law fit scripts (cell + noise scaling)
│   ├── 2026-04-15_14-43_compute_loss_scaling.py
│   ├── 2026-04-15_14-56_plot_loss_scaling.ipynb
│   ├── 2026-04-15_15-30_compute_linear_probe_scaling_curves.py
│   ├── 2026-04-15_15-30_plot_linear_probe_scaling_curves.ipynb
│   ├── 2026-04-16_14-43_*_state_hparam_sweep_pbmc.{py,ipynb}
│   ├── 2026-04-20_10-49_*_ksg_scaling_curves.{py,ipynb}
│   ├── 2026-04-20_14-31_*_state_model_sizing_pbmc.{py,ipynb}
│   ├── 2026-04-21_14-00_*_finetune_pretrained_state_pbmc_noise_scaling.{py,ipynb}
│   ├── 2026-04-22_15-49_*_geneformer_model_sizing_pbmc.{py,ipynb}
│   ├── 2026-04-24_11-49_*_shendure_geneformer_checkpoint_mi.{py,ipynb}
│   ├── 2026-04-08_veryfing_data_correctness/   # Data-completeness checks
│   ├── big_fig.ipynb                  # Main results figure
│   ├── collapse.ipynb                 # Universal-collapse analysis
│   ├── cell_number_scaling.ipynb      # Cell-number scaling curves
│   └── ...
│
└── seq/, images/                       # Supplementary experiments
```

---

## Datasets and algorithms

| Dataset | Description | Signal columns | Sizes | Quality range |
|---------|-------------|----------------|-------|---------------|
| **PBMC** | CITE-seq immune cells | `celltype.l3`, `protein_counts` | 100 -- 100,000 | 0.001 -- 1.0 |
| **larry** | Lineage tracing (Klein lab) | `clone` | 100 -- 100,000 | 0.004 -- 1.0 |
| **merfish** | Spatial transcriptomics | `cur_idx`, `ng_idx` | 100 -- 60,000 | 0.027 -- 1.0 |
| **shendure** | Mouse embryo atlas | `author_day` | 100 -- 10,000,000 | 0.004 -- 1.0 |

| Algorithm | Class | Embedding dim | Key hyperparameters |
|-----------|-------|---------------|---------------------|
| **Geneformer** | `scaling_laws.algo.Geneformer` | 256 | 4 heads, 3 layers, 512 FFN, lr 1e-3, batch 64 |
| **scVI** | `scaling_laws.algo.SCVI` | 16 | 512 hidden, 1 layer, ZINB loss, lr 1e-3 |
| **STATE** | `scaling_laws.algo.State` | 256 | 4 heads, 3 layers, 512 FFN, lr 5e-4 (separate `state` env) |
| **PCA** | `scaling_laws.algo.PCA` | 256 | TruncatedSVD on 750 HVGs |
| **RandomProjection** | `scaling_laws.algo.RandomProjection` | 256 | Gaussian random matrix |

Training epochs scale inversely with dataset size: `max(1, K * 10M / size)` with K=10 for Geneformer/STATE and K=1 for scVI.

---

## Installation

This project ships with both `Geneformer/` and `STATE/state/` vendored in-tree. **No additional clones are needed** — a fresh `git clone` plus the steps below is everything you need on a new machine.

Two conda environments are required because STATE pulls Hydra 1.3, Lightning, and peft 0.11, which conflict with the transformers 4.40 stack used by Geneformer. The orchestrator (`scaling_laws.prepare.data.Experiments`) runs in the `modeling` env and shells out to the `state` env for STATE training.

### 1. Clone

```bash
git clone https://github.com/igor-sadalski/Scaling-up-measurement-noise-scaling-laws.git
cd Scaling-up-measurement-noise-scaling-laws
```

### 2. Create the `modeling` env (Geneformer / scVI / PCA / RP / orchestration)

```bash
conda create -n modeling python=3.10 -y
conda activate modeling

pip install -r requirements.txt          # transformers 4.40, scvi-tools, latentmi, numba, ...
pip install -e Geneformer                # vendored Geneformer (editable)
pip install -e scaling_laws              # core package (editable)

# Verify imports succeed
python -c "import scaling_laws, geneformer; print('modeling env OK')"
```

### 3. Create the `state` env (STATE only)

```bash
conda create -n state python=3.11 -y
conda activate state

pip install -e STATE/state               # installs the `state` CLI

# Verify
python -c "import state; print('state env OK')"
state --help
```

### 4. Configure data location (env vars)

`scaling_laws/paths.py` is the single source of truth for every absolute path used by scripts in this repo. Defaults assume `~/noise_scaling/data` and `~/miniconda3/envs/state`; override via env vars only if your layout differs.

| Variable | Default | Purpose |
|----------|---------|---------|
| `NOISE_SCALING_DATA_DIR` | `~/noise_scaling/data` | Input H5ADs + experiment outputs |
| `NOISE_SCALING_OUTPUT_BASE` | `$NOISE_SCALING_DATA_DIR/other` | Where sweep outputs land |
| `STATE_PYTHON` | `~/miniconda3/envs/state/bin/python` | Python interpreter in the STATE env |
| `STATE_PACKAGE_DIR` | `<repo>/STATE/state` | STATE source dir (override only if not in-tree) |

```bash
# Optional — only set what you need to override:
export NOISE_SCALING_DATA_DIR=/path/to/your/data
export NOISE_SCALING_OUTPUT_BASE="$NOISE_SCALING_DATA_DIR/other"
export STATE_PYTHON="$(conda run -n state which python)"
```

### 5. Sync data from S3

The full mirror is ~6.2 TB. For development you only need the slice your experiment reads:

```bash
# Full mirror (huge):
aws s3 sync s3://measurement-noise-scaling-laws/data/ "$NOISE_SCALING_DATA_DIR"

# Just one PBMC slice (~10 GB, enough to retrain a single hparam config):
aws s3 sync s3://measurement-noise-scaling-laws/data/PBMC/100000/ \
            "$NOISE_SCALING_DATA_DIR/PBMC/100000/"
```

The bucket is public — no AWS credentials are required.

### 6. Final verification

```bash
conda activate modeling
python -m scaling_laws.paths
# Every line should print "[OK ]". A "MISSING" entry usually means you
# either need to set the corresponding env var (step 4) or sync more data.
```

### Important: launch from the `modeling` env

Worker subprocesses (`ProcessPoolExecutor`) inherit `sys.path` from the parent process, so any script that imports from `scaling_laws.*` must be launched from the env where it was installed:

```bash
conda activate modeling
python run_pbmc_whole.py
```

The orchestrator transparently shells out to `STATE_PYTHON` for STATE jobs — you never need to switch envs manually mid-run.

---

## Running the experiments

### Full sweeps (one dataset at a time)

Each `run_{dataset}_whole.py` script does sampling, downsampling, tokenization (Geneformer rank-value encoding), training, embedding, and MI estimation across all (size, quality) combinations:

```bash
conda activate modeling

python run_pbmc_whole.py        # PBMC: 10 sizes x 10 qualities x {Geneformer, scVI, PCA, RP}
python run_larry_whole.py       # larry
python run_merfish_whole.py     # merfish
python run_shendure_whole.py    # shendure (up to 10M cells)
```

On the very first run for a dataset, also run `hvg.ipynb` (or `python prepare_hvg.py`) to pre-compute the 750-gene HVG mask used by the PCA baseline.

### STATE sweeps

STATE has its own runner because it (a) lives in a separate conda env and (b) needs an extra `prepare_state_data()` step before training. Always run the matching base-dataset script first so the H5ADs and metadata exist on disk; the STATE runner reuses those artifacts.

```bash
# After run_shendure_whole.py has populated the data dir:
python run_state_shendure_whole.py

# For other datasets, see the dated drivers under analysis/
# (e.g. analysis/2026-04-14_14-24_run_state_merfish_whole.py
#       analysis/2026-04-15_10-18_run_state_all_datasets.py — runs larry/merfish/shendure).
```

### One-off jobs

`single_job.py` runs a single (dataset, size, quality, algo, seed) experiment:

```bash
python single_job.py \
    --dataset merfish \
    --sizes 7113 \
    --qualities 1.0 \
    --algos Geneformer \
    --max_epochs 1000 \
    --device 0 \
    --seed 42
```

### SLURM

```bash
sbatch run_pbmc_whole.slurm
sbatch run_merfish_whole.slurm
sbatch run_larry_whole.slurm
sbatch run_shendure_whole.slurm
```

---

## Reproducing figures from the paper

Every CSV and PNG under `analysis/final_results/` is produced by a `compute_*.py` (writes the CSV) and then a `plotting_*.ipynb` (reads the CSV and writes the PNG). The table below is the authoritative map. **All paths are relative to the repo root**; all commands assume `conda activate modeling`.

The `compute_*.py` scripts read the trained models / embeddings / MI estimates that the `run_*_whole.py` sweeps have already produced (or that you have synced from `s3://measurement-noise-scaling-laws/data/`). The `plotting_*.ipynb` notebooks read the matching CSV in `analysis/final_results/` and write the PNG back into the same directory.

| Artifact (`analysis/final_results/`) | Compute (`.py`) | Plot (`.ipynb`) | What it shows |
|--------------------------------------|-----------------|-----------------|---------------|
| `cell_scaling.csv` | `analysis/2025-11-18_17-10_hyperparam_fits_cell_scaling.py` | `analysis/cell_number_scaling.ipynb` | Power-law fit of MI vs. dataset size, per (dataset, algo, signal). |
| `noise_scaling.csv` | `analysis/2025-11-18_17-20_hyperparm_fits_noise_scaling.py` | `analysis/2025-11-18_18_visualize_the_results_noise_scaling.py` | Power-law fit of MI vs. quality. |
| `collect_mi_results.{csv,png}` | `analysis/2026-04-08_veryfing_data_correctness/collect_mi_results_from_disk.ipynb` | `analysis/2026-04-08_veryfing_data_correctness/plot_mi_scaling.ipynb` | Disk-walked inventory of MI files + per-(size, quality) curves. |
| `geneformer_loss_scaling.{csv,png}` | `analysis/2026-04-15_14-43_compute_loss_scaling.py` | `analysis/2026-04-15_14-56_plot_loss_scaling.ipynb` | Geneformer pretraining loss vs. cells / quality. |
| `geneformer_loss_curves_sml.{csv,png}` | (rolled into `2026-04-15_14-43_compute_loss_scaling.py`) | `analysis/2026-04-18_09-49_plotting_loss_curves_all_algos.ipynb` | Small/medium/large loss curves overlaid across algos. |
| `linear_probe_scaling.{csv,png}` | `analysis/2026-04-15_15-30_compute_linear_probe_scaling_curves.py` | `analysis/2026-04-15_15-30_plot_linear_probe_scaling_curves.ipynb` | Linear probe accuracy vs. cells, per algo. |
| `hyperpam_sweep.{csv,png}` | `analysis/2026-04-16_14-43_compute_state_hparam_sweep_pbmc.py` | `analysis/2026-04-16_14-43_plotting_state_hparam_sweep_pbmc.ipynb` | STATE hyperparameter sweep on PBMC. |
| `ksg_vs_quality_shendure_SCVI.{csv,png}` | `analysis/2026-04-20_10-49_compute_ksg_scaling_curves.py` | `analysis/2026-04-20_10-49_plotting_ksg_scaling_curves.ipynb` | KSG vs. LMI MI estimator agreement on shendure / scVI. |
| `state_model_size_sweep.{csv,png}` | `analysis/2026-04-20_14-31_compute_state_model_sizing_pbmc.py` | `analysis/2026-04-20_14-31_plotting_state_model_sizing_pbmc.ipynb` | STATE parameter-count sweep on PBMC. |
| `finetune_se_100m_state_pbmc_noise_scaling.{csv,png}` | `analysis/2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py` | `analysis/2026-04-21_14-00_plotting_finetune_pretrained_state_pbmc_noise_scaling.ipynb` | Finetuned 100M-param STATE noise-scaling on PBMC. |
| `geneformer_model_size_sweep.{csv,png}` | `analysis/2026-04-22_15-49_compute_geneformer_model_sizing_pbmc.py` | `analysis/2026-04-22_15-49_plotting_geneformer_model_sizing_pbmc.ipynb` | Geneformer parameter-count sweep on PBMC. |
| `shendure_geneformer_checkpoint_mi.{csv,png}` | `analysis/2026-04-24_11-49_compute_shendure_geneformer_checkpoint_mi.py` | `analysis/2026-04-24_11-49_plotting_shendure_geneformer_checkpoint_mi.ipynb` | MI of public Geneformer checkpoints on shendure. |

### Regenerate any single figure

```bash
conda activate modeling

# Example: regenerate state_model_size_sweep.{csv,png}
python analysis/2026-04-20_14-31_compute_state_model_sizing_pbmc.py
jupyter nbconvert --to notebook --execute --inplace \
    analysis/2026-04-20_14-31_plotting_state_model_sizing_pbmc.ipynb
```

To regenerate **all** aggregate figures from scratch you must first run the full sweeps (`run_pbmc_whole.py`, ..., `run_shendure_whole.py`, `run_state_shendure_whole.py`). To regenerate them from the published outputs without retraining, sync the relevant slices from S3 (step 5 above) and skip straight to the `compute_*.py` script.

---

## Pipeline details

For each (dataset, size, quality, algorithm) combination the orchestrator executes:

1. **Sample** -- draw `num_cells` from the full dataset (deterministic given seed).
2. **Downsample** -- multiplicatively reduce counts by the quality factor (Numba JIT) to simulate measurement noise.
3. **Preprocess** -- normalize, select HVGs, tokenize (Geneformer rank-value encoding), or run STATE preprocessing.
4. **Train** -- fit the embedding model on the train split.
5. **Embed** -- generate latent representations on the held-out test split.
6. **Evaluate** -- estimate MI between embeddings and signal variables using LMI (4 seeds).

Implementation notes:

- `Experiments` (`scaling_laws/src/scaling_laws/prepare/data.py`) is the central orchestrator: handles sampling, downsampling, tokenization, STATE preprocessing, and parallel GPU job scheduling.
- Gene medians for Geneformer rank-value encoding are computed via t-digest (`crick`) for memory efficiency on the 10M-cell shendure sweep.
- `max_epochs` scales inversely with dataset size: `max(1, base_epochs * 10M / size)` so small datasets train longer.
- STATE runs as a subprocess in its own conda env via `STATE_PYTHON`.

---

## Mutual information estimation

MI is computed using the [latentmi](https://github.com/irwin-deng/latentmi) (LMI) estimator with 4 seeds per signal (42, 1404, 2303, 2701). Each estimation produces:

- `lmi_mutual_information.txt` -- scalar MI estimate (nats)
- `lmi_embeddings.npy` -- optimized low-dim projection
- `lmi_model.pt` -- the trained LMI critic network

KSG is used as a sanity-check estimator on a subset of configurations (see the `ksg_vs_quality_*` row in the table above).
