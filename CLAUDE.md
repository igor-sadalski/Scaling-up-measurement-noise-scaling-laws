# Scaling-up-measurement-noise-scaling-laws - Claude Code Instructions

## Overview

This repository studies how single-cell embedding algorithms scale with **dataset size** and **measurement noise** (data quality). It trains 5 embedding methods across 4 datasets at 10 sizes x 10 noise levels, estimates mutual information (MI) between learned representations and biological signals, and fits power-law scaling curves.

**Scale**: 4 datasets x 10 sizes x 10 qualities x 5 algorithms = 2,000 training runs; x 2-4 signals x 4 MI seeds = ~40,000 MI estimates.

## Environment

Activate the conda environment before running Python commands:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling
```

**Exception**: STATE algorithm requires its own environment at `/home/igor/miniconda3/envs/state/bin/python` and the STATE package at `/home/igor/noise_scaling/modeling/STATE/state`.

## Package Structure

Two local packages installed in editable mode:
- `scaling_laws/` -- core package (`pip install -e .`)
- `Geneformer/` -- Geneformer library (`pip install -e .`)

```
scaling_laws/src/scaling_laws/
├── prepare/
│   ├── data.py          # PrepareData (single experiment) + Experiments (orchestrator)
│   └── utils.py         # Numba-optimized downsampling, t-digest medians
├── algo/
│   ├── abc.py           # BaseAlgorithm: abstract base with MI estimation (latentmi)
│   ├── geneformer.py    # Geneformer: masked LM, 256-dim, HuggingFace Trainer
│   ├── scvi.py          # SCVI: VAE, 16-dim, scvi-tools
│   ├── state.py         # State: self-supervised transformer, 256-dim, CLI subprocess
│   ├── pca.py           # PCA: TruncatedSVD on HVGs, 256-dim
│   ├── rp.py            # RandomProjection: Gaussian RP baseline, 256-dim
│   └── utils.py         # Shared: normalize_adata, HVG selection, PCATransformer
├── h5ad_reader.py       # Memory-efficient chunked H5AD reader
└── s3_retriever.py      # Unified S3/local data loader + result collection (860 lines)
```

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `run_pbmc_whole.py` | Full PBMC experiment (10 sizes x 10 qualities x 4 algos) |
| `run_larry_whole.py` | Full larry lineage experiment |
| `run_merfish_whole.py` | Full MERFISH spatial experiment |
| `run_shendure_whole.py` | Full shendure embryo atlas experiment |
| `run_state_merfish_all.py` | STATE on all merfish configs |
| `single_job.py` | CLI for a single (dataset, size, quality, algo) experiment |
| `prepare_hvg.py` | Pre-compute HVG masks (750 genes) |
| `run_state_test.py` | Quick STATE end-to-end test |

## Running Experiments

### Full sweep (one dataset)
```bash
python run_pbmc_whole.py
```
This downloads raw data, samples at 10 sizes, downsamples at 10 qualities, tokenizes, then trains/embeds/computes MI in parallel across GPUs.

### Single experiment
```bash
python single_job.py --dataset merfish --sizes 7113 --qualities 1.0 --algos Geneformer --device 0
```

### SLURM submission
```bash
sbatch run_pbmc_whole.slurm
```

## Pipeline Stages

```
Raw h5ad  ->  Sample N cells  ->  Downsample by quality factor  ->  Tokenize (Geneformer)
                                                                 ->  Create CSV manifests (STATE)
          ->  Train algorithm  ->  Extract test embeddings  ->  Estimate MI (4 seeds)
          ->  Fit power laws: MI ~ size^alpha, MI ~ quality^beta
```

## Algorithms

| Algorithm | Class | Dim | Key Config |
|-----------|-------|-----|------------|
| Geneformer | `algo/geneformer.py` | 256 | 256 hidden, 4 heads, 3 layers, 512 FFN, lr=1e-3, batch=64 |
| scVI | `algo/scvi.py` | 16 | 512 hidden, 1 layer, ZINB loss, lr=1e-3 |
| STATE | `algo/state.py` | 256 | 256 hidden, 4 heads, 3 layers, 512 FFN, lr=5e-4 |
| PCA | `algo/pca.py` | 256 | TruncatedSVD on 750 HVGs |
| RandomProjection | `algo/rp.py` | 256 | GaussianRandomProjection |

## Datasets

| Dataset | Signal Columns | Max Cells | Sizes |
|---------|---------------|-----------|-------|
| PBMC | celltype.l3, protein_counts | 100k | 100 to 100k (10 log-spaced) |
| larry | clone | 100k | 100 to 100k |
| merfish | cur_idx, ng_idx | 60k | 100 to 60k |
| shendure | author_day | 10M | 100 to 10M |

Qualities: 10 log-spaced values from ~0.001 to 1.0 (dataset-specific lower bounds).

## MI Estimation

Uses `latentmi` (LMI estimator) with 4 seeds (42, 1404, 2303, 2701) per signal. Results saved as:
- `lmi_mutual_information.txt` -- scalar MI value (nats)
- `lmi_embeddings.npy` -- optimized low-dim projection
- `lmi_model.pt` -- trained LMI network

## Data Location

All data at `/home/igor/noise_scaling/data/` (symlink to `/opt/dlami/nvme/data`), also on S3:
```bash
aws s3 ls s3://measurement-noise-scaling-laws/data/
```

Use `S3Retriever` for programmatic access:
```python
from scaling_laws.s3_retriever import S3Retriever
s3 = S3Retriever("/home/igor/noise_scaling/data")
embeddings = s3.load_embeddings("PBMC", 10000, 1.0, "Geneformer")
mi = s3.load_mutual_information("PBMC", 10000, 1.0, "Geneformer", "celltype.l3", seed=42)
```

## Analysis

Analysis notebooks and scripts in `analysis/`:
- `big_fig.ipynb`, `big_fig_2.ipynb` -- main results figures
- `collapse.ipynb` -- universal scaling law collapse
- `cell_number_scaling.ipynb` -- size scaling curves
- `comparisons.ipynb` -- algorithm comparisons
- `2025-11-18_*` -- power-law fitting scripts (output: `final_results/cell_scaling.csv`, `noise_scaling.csv`)
- `2026-04-08_veryfing_data_correctness/` -- data validation and completeness checks

## Important Implementation Details

- `Experiments` class (`prepare/data.py`) is the main orchestrator: handles sampling, downsampling, tokenization, parallel GPU job scheduling
- `max_epochs` scales inversely with dataset size: `max(1, base * 10M / size)`
- Downsampling uses Numba-JIT for sub-second processing of millions of cells
- Gene medians computed with t-digest (crick library) for memory efficiency
- Geneformer tokenization uses rank-value encoding (genes ordered by expression rank)
- STATE runs as a subprocess with its own conda environment
