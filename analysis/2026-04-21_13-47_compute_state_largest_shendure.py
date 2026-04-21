"""Re-run STATE on the 3 largest shendure sizes across all 10 qualities.

Background: the original full sweep
(``analysis/2026-04-16_14-49_compute_state_all_datasets.py``) trained STATE on
the full shendure size grid before the "≥1 epoch of data" fix landed. The 3
largest sizes (774263, 2782559, 10000000) are the ones that suffered from the
bug — at those scales the 15k step budget could finish before consuming a full
epoch, so the resulting models are not comparable to the smaller-size runs.

This script repeats the STATE training/embedding/MI pipeline for just those 3
sizes × all 10 qualities × seed=42, overwriting the existing State results in
place (``retrain=True``, ``reembed=True``, ``recompute_mutual_information=True``).
Other datasets and the smaller shendure sizes are intentionally untouched.

Training protocol mirrors the original sweep (validation every 1000 steps,
early-stop patience=5, 15k optimizer-step budget) plus the new ≥1 epoch
guarantee from ``scaling_laws.algo.state.State``.
"""

import sys
from pathlib import Path

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


_log_f = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
print(f"Logging to {LOG_PATH}")

# ── Imports ───────────────────────────────────────────────────────────────
from scaling_laws.prepare.data import Experiments

from scaling_laws.paths import DATA_DIR
path_to_data_dir = str(DATA_DIR)

# Only the 3 largest shendure sizes — the ones the ≥1-epoch fix is supposed
# to repair. All 10 qualities, single seed=42 (same as the original sweep).
DATASETS = {
    "shendure": {
        "sizes": [774263, 2782559, 10000000],
        "qualities": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        "signal_columns": ["author_day"],
        "seeds": [42],
        "jobs_per_gpu": 2,
    },
}

for ds_name, cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  {len(cfg['sizes'])} sizes x {len(cfg['qualities'])} qualities "
          f"x {len(cfg['seeds'])} seeds = {len(cfg['sizes']) * len(cfg['qualities']) * len(cfg['seeds'])} runs")
    print(f"  sizes={cfg['sizes']}")
    print(f"  jobs_per_gpu={cfg['jobs_per_gpu']}")
    print(f"  Validation every 1000 steps, early stopping patience=5, max_steps=15000 (no max_epochs cap), saving train/val/test_loss.txt")
    print(f"{'='*70}\n")

    # ── Train / embed / MI ────────────────────────────────────────────
    for seed in cfg["seeds"]:
        print(f"\n[{ds_name}] Training + MI with seed={seed}...")
        experiments = Experiments(
            datasets=[ds_name],
            sizes=cfg["sizes"],
            qualities=cfg["qualities"],
            algos=["State"],
            path_to_data_dir=path_to_data_dir,
            signal_columns=cfg["signal_columns"],
            device=0,
            seed=seed,
        )
        experiments.parallel_run(
            sleep_time=0.2,
            retrain=True,
            reembed=True,
            recompute_mutual_information=True,
            jobs_per_gpu=cfg["jobs_per_gpu"],
            log_dir=f"{path_to_data_dir}/{ds_name}/logs/state_run_largest_seed_{seed}",
        )

    print(f"\n[{ds_name}] Done.")

print("\n" + "=" * 70)
print("  LARGEST SHENDURE SIZES COMPLETE (774263, 2782559, 10000000)")
print("=" * 70)
