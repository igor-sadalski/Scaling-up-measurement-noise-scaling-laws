# Scaling Up Measurement Noise Scaling Laws

This repository studies how single-cell embedding algorithms scale with **dataset size** and **measurement noise** (data quality). We systematically train and evaluate multiple embedding methods across four single-cell datasets at 10 logarithmically-spaced sizes and 10 noise levels, then estimate mutual information (MI) between learned representations and biologically meaningful signal variables.

## Data

All experimental outputs (trained models, embeddings, MI estimates) are publicly available on S3:

```
s3://measurement-noise-scaling-laws/data/
```

No credentials are required. See the [data README](../../data/README.md) for full documentation and worked examples.

```python
from scaling_laws.s3_retriever import S3Retriever

data = S3Retriever()  # defaults to the public S3 bucket
emb = data.load_embeddings("merfish", num_cells=7113, quality=1.0, algorithm="Geneformer")
mi  = data.load_mutual_information("merfish", num_cells=7113, quality=1.0,
                                    algorithm="Geneformer", signal="ng_idx", seed=42)
```

## Repository structure

```
.
├── scaling_laws/                    # Core Python package (pip install -e .)
│   └── src/scaling_laws/
│       ├── prepare/
│       │   ├── data.py              # Experiments & PrepareData classes
│       │   └── utils.py             # Numba-optimized downsampling kernels
│       ├── algo/
│       │   ├── abc.py               # BaseAlgorithm interface
│       │   ├── geneformer.py        # Geneformer (masked LM)
│       │   ├── scvi.py              # scVI (VAE)
│       │   ├── state.py             # STATE (self-supervised transformer)
│       │   ├── pca.py               # PCA baseline
│       │   ├── rp.py                # Random projection baseline
│       │   └── utils.py             # Shared helpers (HVG selection, normalization)
│       ├── h5ad_reader.py           # Memory-efficient chunked H5AD reader
│       └── s3_retriever.py          # S3Retriever: load data/models/MI from S3 or local
│
├── Geneformer/                      # Geneformer library (installed as editable package)
│   └── geneformer/
│       ├── tokenizer.py             # RNA-seq tokenization
│       ├── pretrainer.py            # Masked language model pre-training
│       ├── emb_extractor.py         # Embedding extraction
│       └── ...
│
├── analysis/                        # Post-hoc analysis notebooks and scripts
│   ├── big_fig.ipynb                # Main results figure
│   ├── collapse.ipynb               # Scaling law collapse analysis
│   ├── cell_number_scaling.ipynb    # Cell-number scaling curves
│   ├── 2025-11-18_*_hyperparam_fits*.py   # Power-law fitting
│   ├── 2025-11-18_*_visualize*.py         # Visualization
│   ├── final_results/               # Aggregated CSVs
│   └── ...
│
├── seq/                             # GISAID sequence model experiments
│   ├── train_gisaid_model.py
│   ├── gisaid_analysis.ipynb
│   └── ...
│
├── images/                          # Supplementary experiments (kidney, tissueMNIST)
│
├── run_pbmc_whole.py                # Full PBMC experiment sweep
├── run_larry_whole.py               # Full larry experiment sweep
├── run_merfish_whole.py             # Full merfish experiment sweep
├── run_shendure_whole.py            # Full shendure experiment sweep
├── run_state_merfish_all.py         # STATE on all merfish configs
├── run_state_small_merfish.py       # STATE on small merfish (multi-seed)
├── run_state_test.py                # Quick STATE end-to-end test
├── single_job.py                    # CLI: run a single (dataset, size, quality, algo) job
├── prepare_hvg.py                   # Pre-compute highly variable gene masks
├── hvg.ipynb                        # HVG computation notebook
│
├── *.slurm                          # SLURM job submission scripts
├── install_dependencies.sh          # Automated dependency installation
└── requirements.txt                 # Python dependencies
```

## Datasets

| Dataset | Description | Signal columns | Sizes | Quality range |
|---------|------------|----------------|-------|---------------|
| **PBMC** | CITE-seq immune cells | `celltype.l3`, `protein_counts` | 100 -- 100,000 | 0.001 -- 1.0 |
| **larry** | Lineage tracing (Klein lab) | `clone` | 100 -- 100,000 | 0.004 -- 1.0 |
| **merfish** | Spatial transcriptomics | `cur_idx`, `ng_idx` | 100 -- 60,000 | 0.027 -- 1.0 |
| **shendure** | Embryo atlas (cellxgene) | `author_day` | 100 -- 10,000,000 | 0.004 -- 1.0 |

## Embedding algorithms

| Algorithm | Class | Embedding dim | Key hyperparameters |
|-----------|-------|---------------|---------------------|
| **Geneformer** | `scaling_laws.algo.Geneformer` | 256 | 4 heads, 3 layers, 512 FFN, LR 1e-3 |
| **scVI** | `scaling_laws.algo.SCVI` | 16 | 512 hidden, 1 layer, ZINB loss |
| **STATE** | `scaling_laws.algo.State` | 256 | 4 heads, 3 layers, 512 FFN |
| **PCA** | `scaling_laws.algo.PCA` | 256 | HVG-selected features |
| **RandomProjection** | `scaling_laws.algo.RandomProjection` | 256 | Gaussian random matrix |

Training epochs scale inversely with dataset size: `max(1, K * 10M / size)` where K=10 for Geneformer/STATE and K=1 for scVI.

## Installation

```bash
# Clone the repository
git clone https://github.com/igor-sadalski/Scaling-up-measurement-noise-scaling-laws.git
cd Scaling-up-measurement-noise-scaling-laws

# Option A: automated install
bash install_dependencies.sh

# Option B: manual install
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd Geneformer && pip install -e . && cd ..
cd scaling_laws && pip install -e . && cd ..
```

For STATE experiments, an additional conda environment (`state`) with the [Arc Institute STATE package](https://github.com/ArcInstitute/state) is required.

## Running experiments

### Full dataset sweep

Each `run_{dataset}_whole.py` script trains all algorithms across all (size, quality) combinations:

```bash
python run_pbmc_whole.py
python run_larry_whole.py
python run_merfish_whole.py
python run_shendure_whole.py
```

On first run, uncomment the tokenization code in each script and run `hvg.ipynb` to pre-compute HVGs.

### Single experiment

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

### SLURM submission

```bash
sbatch run_merfish_whole.slurm
```

## Experiment pipeline

For each (dataset, size, quality, algorithm) combination:

1. **Sample** -- draw `num_cells` from the full dataset
2. **Downsample** -- reduce counts by the quality factor to simulate measurement noise
3. **Preprocess** -- normalize, tokenize (Geneformer), or prepare inputs
4. **Train** -- fit the embedding model on the training subset
5. **Embed** -- generate latent representations for the held-out test set
6. **Evaluate** -- estimate MI between embeddings and signal variables using LMI (4 seeds)

## Mutual information estimation

MI is computed using the [latentmi](https://github.com/irwin-deng/latentmi) package (LMI estimator) with 4 random seeds (42, 1404, 2303, 2701) per experiment. Each run produces:

- `lmi_mutual_information.txt` -- scalar MI in nats
- `lmi_embeddings.npy` -- optimized low-dimensional projection
- `lmi_model.pt` -- trained LMI network

## Key dependencies

- PyTorch >= 2.0, transformers 4.40, scvi-tools
- anndata, scanpy, datasets (HuggingFace)
- latentmi (MI estimation)
- numba (optimized downsampling)
- wandb (experiment tracking)

See `requirements.txt` for the full list.
