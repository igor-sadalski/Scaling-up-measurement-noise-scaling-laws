"""
Train a STATE Embedding (SE) model from scratch on preprocessed.h5ad.

SE is self-supervised: it learns cell embeddings by predicting gene expression
patterns. No perturbation labels needed.

Steps:
  1. Split data into train/val h5ad files
  2. Create CSV manifests for STATE
  3. Run `state emb preprocess` (creates one-hot gene embeddings + mappings)
  4. Run `state emb fit` (trains the SE transformer)

Usage:
    conda activate state
    python train_state.py
"""

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TRAIN_H5AD = BASE_DIR / "data" / "raw_train_data.h5ad"
VAL_H5AD = BASE_DIR / "data" / "raw_val.h5ad"
TEST_H5AD = BASE_DIR / "data" / "raw_test_data.h5ad"
TRAIN_CSV = BASE_DIR / "data" / "train.csv"
VAL_CSV = BASE_DIR / "data" / "val.csv"
EMB_OUTPUT_DIR = BASE_DIR / "se_profile"
CONFIG_PATH = BASE_DIR / "se_config.yaml"
CKPT_DIR = BASE_DIR / "se_checkpoints"
PROFILE_NAME = "spatial"

GPU_ID = "2"
STATE_DIR = BASE_DIR / "state"  # where the state package source lives

# ── Step 1: Check train/val data ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: Using existing train/val splits")
print("=" * 60)

adata_train = ad.read_h5ad(TRAIN_H5AD, backed='r')
adata_val = ad.read_h5ad(VAL_H5AD, backed='r')
adata_test = ad.read_h5ad(TEST_H5AD, backed='r')
print(f"  Train: {adata_train.shape[0]} cells x {adata_train.shape[1]} genes <- {TRAIN_H5AD}")
print(f"  Val:   {adata_val.shape[0]} cells x {adata_val.shape[1]} genes <- {VAL_H5AD}")
print(f"  Test:  {adata_test.shape[0]} cells x {adata_test.shape[1]} genes <- {TEST_H5AD}")

# ── Step 2: Create CSV manifests ──────────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Creating CSV manifests")
print("=" * 60)

TRAIN_CSV.write_text(f"species,path,names\nhuman,{TRAIN_H5AD},spatial_train\n")
VAL_CSV.write_text(f"species,path,names\nhuman,{VAL_H5AD},spatial_val\n")
print(f"  Train CSV: {TRAIN_CSV}")
print(f"  Val CSV:   {VAL_CSV}")

# ── Step 3: Copy default config and run preprocess ────────────────────────────
print()
print("=" * 60)
print("Step 3: Preprocessing (gene embeddings + mappings)")
print("=" * 60)

# Copy the default config as our base
import shutil
default_config = STATE_DIR / "src" / "state" / "configs" / "state-defaults.yaml"
shutil.copy(default_config, CONFIG_PATH)
print(f"  Copied base config to {CONFIG_PATH}")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU_ID

cmd_preprocess = [
    sys.executable, "-m", "state", "emb", "preprocess",
    "--profile-name", PROFILE_NAME,
    "--train-csv", str(TRAIN_CSV),
    "--val-csv", str(VAL_CSV),
    "--output-dir", str(EMB_OUTPUT_DIR),
    "--config-file", str(CONFIG_PATH),
    # No --all-embeddings → uses one-hot (no ESM)
]
print(f"  Running: {' '.join(cmd_preprocess)}")
result = subprocess.run(cmd_preprocess, cwd=str(STATE_DIR), env=env)
if result.returncode != 0:
    print("ERROR: Preprocessing failed!")
    sys.exit(result.returncode)

# ── Step 4: Patch config for our small dataset ────────────────────────────────
print()
print("=" * 60)
print("Step 4: Patching config for small dataset")
print("=" * 60)

from omegaconf import OmegaConf

cfg = OmegaConf.load(CONFIG_PATH)

# Point to our profile
cfg.embeddings.current = PROFILE_NAME
cfg.dataset.current = PROFILE_NAME

# Adjust for small dataset
cfg.dataset.num_cells = int(adata_train.shape[0])
cfg.dataset.num_train_workers = 4
cfg.dataset.num_val_workers = 2

# Model: use the one-hot embedding size (= number of genes)
# The preprocess step already set embeddings.spatial.size to n_genes

# Training config
cfg.experiment.num_epochs = 10
cfg.experiment.num_gpus_per_node = 1
cfg.experiment.num_nodes = 1
cfg.experiment.val_check_interval = 100
cfg.experiment.limit_val_batches = 50
cfg.experiment.checkpoint.path = str(CKPT_DIR)
cfg.experiment.checkpoint.every_n_train_steps = 500
cfg.experiment.checkpoint.monitor = "validation/val_loss"
cfg.experiment.checkpoint.save_top_k = 1

# Match Geneformer architecture: 256 hidden, 4 heads, 3 layers, 512 context
cfg.model.batch_size = 64
cfg.model.emsize = 256       # d_model — matches Geneformer hidden_size
cfg.model.d_hid = 512        # feedforward — matches Geneformer intermediate_size
cfg.model.nhead = 4          # attention heads — matches Geneformer
cfg.model.nlayers = 3        # transformer layers — matches Geneformer
cfg.model.output_dim = 256   # cell embedding dim — matches hidden_size
cfg.model.dataset_correction = False  # only 1 dataset, no need
cfg.model.dropout = 0.02     # matches Geneformer dropout

# Context window 512 — matches Geneformer max_position_embeddings
cfg.dataset.pad_length = 512
cfg.dataset.P = 128
cfg.dataset.N = 128
cfg.dataset.S = 128

# Optimizer — similar to Geneformer
cfg.optimizer.max_lr = 1e-3
cfg.optimizer.gradient_accumulation_steps = 1
cfg.optimizer.weight_decay = 0.001

# Disable wandb
cfg.wandb.enable = False

# Disable validations that require specific datasets we don't have
cfg.validations.diff_exp.enable = False
cfg.validations.perturbation.enable = False

OmegaConf.save(cfg, CONFIG_PATH)
print(f"  Saved patched config to {CONFIG_PATH}")
print(f"  embedding size: {cfg.embeddings[PROFILE_NAME].size}")
print(f"  batch_size: {cfg.model.batch_size}")
print(f"  num_epochs: {cfg.experiment.num_epochs}")
print(f"  num_cells: {cfg.dataset.num_cells}")

# ── Step 5: Train SE model ───────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 5: Training SE model")
print("=" * 60)

import time

cmd_fit = [
    sys.executable, "-m", "state", "emb", "fit",
    "--conf", str(CONFIG_PATH),
]
print(f"  Running: {' '.join(cmd_fit)}")
print()

t_start = time.time()
result = subprocess.run(cmd_fit, cwd=str(STATE_DIR), env=env)
t_elapsed = time.time() - t_start

if result.returncode != 0:
    print("ERROR: Training failed!")
    sys.exit(result.returncode)

print(f"\n  Training time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min) for {cfg.experiment.num_epochs} epochs")
print(f"  Per epoch: {t_elapsed/cfg.experiment.num_epochs:.1f}s")

# ── Step 6: Embed test set and plot UMAP ─────────────────────────────────────
print()
print("=" * 60)
print("Step 6: Embedding test set + UMAP")
print("=" * 60)

import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

# Find best checkpoint — search all subdirs for last.ckpt
ckpt_subdirs = sorted(glob.glob(os.path.join(str(CKPT_DIR), "vci_pretrain_*")))
ckpt_dir_pattern = ckpt_subdirs[-1] if ckpt_subdirs else str(CKPT_DIR)
last_ckpt = os.path.join(ckpt_dir_pattern, "last.ckpt")
if os.path.exists(last_ckpt):
    ckpt_path = last_ckpt
else:
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir_pattern, "*.ckpt")))
    ckpt_path = ckpt_files[-1]
print(f"  Using checkpoint: {ckpt_path}")

from state.emb.inference import Inference

inf = Inference()
inf.load_model(ckpt_path)

OUTPUT_H5AD = BASE_DIR / "data" / "test_embedded.h5ad"
OUTPUT_PLOT = BASE_DIR / "umap_test_embeddings.png"

embeddings = inf.encode_adata(
    input_adata_path=str(TEST_H5AD),
    output_adata_path=str(OUTPUT_H5AD),
    emb_key="X_state",
    dataset_name="spatial_test",
)
print(f"  Embeddings shape: {embeddings.shape}")

# UMAP
print("  Running UMAP...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
umap_coords = reducer.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=3, alpha=0.6, color="#4a90d9", rasterized=True)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("SE Model - Test Set Embeddings")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")

# Save UMAP coords into the h5ad
adata_out = ad.read_h5ad(OUTPUT_H5AD)
adata_out.obsm["X_umap_state"] = umap_coords
adata_out.write_h5ad(OUTPUT_H5AD)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Checkpoint:    {ckpt_path}")
print(f"  Embedded h5ad: {OUTPUT_H5AD}")
print(f"  UMAP plot:     {OUTPUT_PLOT}")
