"""
Run STATE on small MERFISH examples (sizes 100, 414) with long training
(similar conditions as Geneformer), compute embeddings + MI, and compare
with other algorithms.

Usage:
    conda activate modeling
    python run_state_small_merfish.py
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from scaling_laws.prepare.data import Experiments

DATA_DIR = "/mnt/nvme/noise_laws/data"
DATASET = "merfish"
SIGNAL_COLUMNS = ["cur_idx", "ng_idx"]
SEEDS = [42, 1404, 2701]

# Two small sizes, quality=1.0 (no noise) — matching Geneformer conditions
RUNS = [
    {"size": 100,  "quality": 1.0, "max_epochs": 5000, "device": 5},
    {"size": 414,  "quality": 1.0, "max_epochs": 2000, "device": 6},
]

ALL_ALGOS = ["Geneformer", "SCVI", "PCA", "RandomProjection", "State"]

# ── 1. Train STATE + embed + compute MI ──────────────────────────────────

for run in RUNS:
    for seed in SEEDS:
        size, quality = run["size"], run["quality"]
        max_epochs, device = run["max_epochs"], run["device"]

        print("=" * 70)
        print(f"STATE | size={size}, quality={quality}, seed={seed}, "
              f"max_epochs={max_epochs}, GPU={device}")
        print("=" * 70)

        experiments = Experiments(
            datasets=[DATASET],
            sizes=[size],
            qualities=[quality],
            algos=["State"],
            path_to_data_dir=DATA_DIR,
            signal_columns=SIGNAL_COLUMNS,
            seed=seed,
        )

        t0 = time.time()
        experiments.single_job(
            dataset=DATASET,
            size=size,
            quality=quality,
            algo="State",
            max_epochs=max_epochs,
            early_stopping_patience=5,
            device=device,
            retrain=True,
            reembed=True,
            recompute_mutual_information=True,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

# ── 2. Collect and compare MI results ────────────────────────────────────

print("\n" + "=" * 70)
print("MI COMPARISON: State vs other algorithms")
print("=" * 70)

rows = []
for run in RUNS:
    size, quality = run["size"], run["quality"]
    base = Path(DATA_DIR) / DATASET / str(size) / str(quality) / "results"

    for algo in ALL_ALGOS:
        for seed in SEEDS + [2303]:
            mi_dir = base / algo / "model" / "MI" / str(seed)
            if not mi_dir.exists():
                continue
            for sig_dir in sorted(mi_dir.iterdir()):
                mi_file = sig_dir / "lmi_mutual_information.txt"
                if mi_file.exists():
                    mi_val = float(mi_file.read_text().strip())
                    rows.append({
                        "size": size,
                        "quality": quality,
                        "algo": algo,
                        "seed": seed,
                        "signal": sig_dir.name,
                        "MI": mi_val,
                    })

if rows:
    df = pd.DataFrame(rows)
    # Average across seeds
    summary = (
        df.groupby(["size", "quality", "algo", "signal"])["MI"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["size", "signal", "mean"], ascending=[True, True, False])
    )
    print("\nMI results (averaged across seeds):")
    print(summary.to_string(index=False))

    # Save
    out_path = Path(DATA_DIR) / DATASET / "state_vs_others_small_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")
else:
    print("No MI results found to compare.")

# ── 3. Quick embedding shape/stats comparison ────────────────────────────

print("\n" + "=" * 70)
print("EMBEDDING STATISTICS")
print("=" * 70)

for run in RUNS:
    size, quality = run["size"], run["quality"]
    base = Path(DATA_DIR) / DATASET / str(size) / str(quality) / "results"
    print(f"\n--- Size={size}, Quality={quality} ---")

    for algo in ALL_ALGOS:
        emb_path = base / algo / "model" / "embeddings.csv"
        if emb_path.exists():
            emb = pd.read_csv(emb_path)
            print(f"  {algo:20s}: shape={emb.shape}, "
                  f"mean={emb.values.mean():.4f}, "
                  f"std={emb.values.std():.4f}, "
                  f"min={emb.values.min():.4f}, "
                  f"max={emb.values.max():.4f}")
        else:
            print(f"  {algo:20s}: no embeddings")

print("\nDone!")
