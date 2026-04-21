"""Compute test loss for all existing Geneformer, SCVI, and State experiments.

STATE: reads metrics.csv (no GPU).
Geneformer & SCVI: loads model + forward pass on test set (GPU).
Skips any config where test_loss.txt already exists or model is missing.
"""
import subprocess
import sys
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from scaling_laws.s3_retriever import DATASET_SIZES, DATASET_QUALITIES

from scaling_laws.paths import DATA_DIR
DATASETS = ["PBMC", "larry", "merfish", "shendure"]
NUM_GPUS = 8
WORKERS = 8
SINGLE_JOB = Path(__file__).parent / "single_job.py"


def run_one(ds, sz, q, algo, gpu):
    """Run single_job.py for one (dataset, size, quality, algo) on a given GPU."""
    cmd = [
        sys.executable, str(SINGLE_JOB),
        "--dataset", ds,
        "--sizes", str(sz),
        "--qualities", str(q),
        "--algos", algo,
        "--base_dir", str(DATA_DIR),
        "--device", str(gpu),
        "--max_epochs", "1",
        "--early_stopping_patience", "1",
        "--retrain", "false",
        "--reembed", "false",
        "--recompute_mutual_information", "false",
        "--recompute_loss", "true",
        "--seed", "42",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            return f"FAIL {algo} {ds}/{sz}/{q}: {result.stderr[-200:]}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {algo} {ds}/{sz}/{q}"
    except Exception as e:
        return f"ERROR {algo} {ds}/{sz}/{q}: {e}"
    return None


# ── Step 1: STATE (fast, no GPU, inline) ──────────────────────────────────
print("=" * 60)
print("Step 1: STATE — best val loss from metrics.csv")
print("=" * 60)

done, skip = 0, 0
for ds in DATASETS:
    for sz, q in product(DATASET_SIZES[ds], DATASET_QUALITIES[ds]):
        loss_path = DATA_DIR / ds / str(sz) / str(q) / "results" / "State" / "model" / "test_loss.txt"
        metrics_path = DATA_DIR / ds / str(sz) / str(q) / "results" / "State" / "model" / "loss" / "metrics.csv"
        if loss_path.exists():
            skip += 1
            continue
        if not metrics_path.exists():
            skip += 1
            continue
        try:
            df = pd.read_csv(metrics_path)
            val = df["validation/val_loss"].dropna()
            if val.empty:
                skip += 1
                continue
            loss_path.parent.mkdir(parents=True, exist_ok=True)
            loss_path.write_text(f"{float(val.min()):.6f}")
            done += 1
        except Exception as e:
            print(f"  FAIL State {ds}/{sz}/{q}: {e}")
            skip += 1

print(f"STATE: {done} written, {skip} skipped")

# ── Step 2: Geneformer & SCVI (GPU, subprocess pool) ─────────────────────
print("\n" + "=" * 60)
print("Step 2: Geneformer & SCVI — forward pass on test set")
print("=" * 60)

jobs = []
for ds in DATASETS:
    for sz, q, algo in product(DATASET_SIZES[ds], DATASET_QUALITIES[ds], ["Geneformer", "SCVI"]):
        loss_path = DATA_DIR / ds / str(sz) / str(q) / "results" / algo / "model" / "test_loss.txt"
        if loss_path.exists():
            continue
        # Check model exists
        model_dir = DATA_DIR / ds / str(sz) / str(q) / "results" / algo / "model"
        if algo == "Geneformer" and not (model_dir / "model.safetensors").exists():
            continue
        if algo == "SCVI" and not (model_dir / "model.pt").exists():
            continue
        jobs.append((ds, sz, q, algo))

print(f"Jobs to run: {len(jobs)} (skipped existing/missing)")

failed = []
with ProcessPoolExecutor(max_workers=WORKERS) as executor:
    futures = {}
    for i, (ds, sz, q, algo) in enumerate(jobs):
        gpu = i % NUM_GPUS
        f = executor.submit(run_one, ds, sz, q, algo, gpu)
        futures[f] = (ds, sz, q, algo)

    for f in tqdm(as_completed(futures), total=len(futures), desc="Loss computation"):
        err = f.result()
        if err:
            failed.append(err)
            tqdm.write(err)

print(f"\nDone! {len(jobs) - len(failed)} succeeded, {len(failed)} failed")
for msg in failed:
    print(f"  {msg}")
