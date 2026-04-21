"""
End-to-end test for the State algorithm integration.

Tests the full pipeline: train → embed → LMI on MERFISH data.
Uses existing preprocessed data at $NOISE_SCALING_DATA_DIR/merfish/.

Usage:
    conda activate modeling
    python run_state_test.py
"""

import time
from scaling_laws.prepare.data import Experiments
from scaling_laws.paths import DATA_DIR as _DATA_DIR

DATA_DIR = str(_DATA_DIR)
DATASET = "merfish"
SIZE = 414
QUALITY = 1.0
DEVICE = 0  # GPU to use

print("=" * 60)
print("State Algorithm - End-to-End Test")
print("=" * 60)
print(f"  Dataset: {DATASET}")
print(f"  Size: {SIZE}")
print(f"  Quality: {QUALITY}")
print(f"  GPU: {DEVICE}")
print()

experiments = Experiments(
    datasets=[DATASET],
    sizes=[SIZE],
    qualities=[QUALITY],
    algos=["State"],
    path_to_data_dir=DATA_DIR,
    signal_columns=["cur_idx"],
    device=DEVICE,
)

t_start = time.time()

experiments.single_job(
    dataset=DATASET,
    size=SIZE,
    quality=QUALITY,
    algo="State",
    max_epochs=10,
    early_stopping_patience=3,
    device=DEVICE,
    retrain=True,
    reembed=True,
    recompute_mutual_information=True,
)

t_elapsed = time.time() - t_start

print()
print("=" * 60)
print(f"Test completed in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
print("=" * 60)

# Verify output files
from pathlib import Path
base = Path(DATA_DIR) / DATASET / str(SIZE) / str(QUALITY)
emb_path = base / "results" / "State" / "model" / "embeddings.csv"
print(f"  Embeddings: {emb_path} (exists={emb_path.exists()})")

mi_dir = base / "results" / "State" / "model" / "MI"
if mi_dir.exists():
    for mi_file in sorted(mi_dir.rglob("lmi_mutual_information.txt")):
        with open(mi_file) as f:
            mi_val = f.read().strip()
        print(f"  LMI: {mi_file.relative_to(mi_dir)} = {mi_val}")
