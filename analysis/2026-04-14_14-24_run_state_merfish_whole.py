"""Run STATE on MERFISH (all sizes x all qualities).

Assumes data is already prepared (sampled, downsampled, tokenized)
by run_merfish_whole.py. Only runs STATE preprocessing + training.

Hyperparameters from notebook 2026-04-13_20-36:
  - max_epochs auto-scaled: max(1, 10 * 10M / size), early_stopping_patience=5
  - pad_length=2048, emsize=256, 4 heads, 3 layers, 512 FFN
  - max_lr=1e-4, dropout=0.1, batch_size=64
  - LMI max_epochs=300
"""

import sys
import shutil
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
import numpy as np

datasets = ["merfish"]
# Single point matching notebook 2026-04-13_20-36
sizes = [60000]
qualities = [1.0]
path_to_data_dir = "/home/igor/noise_scaling/data"
signal_columns = ["cur_idx", "ng_idx"]
seeds = [42]

# ── 0. Clear old State results ───────────────────────────────────────────

for size in sizes:
    for quality in qualities:
        state_dir = Path(path_to_data_dir) / "merfish" / str(size) / str(quality) / "results" / "State"
        if state_dir.exists():
            shutil.rmtree(state_dir)
            print(f"Removed {state_dir}")

# ── 1. STATE preprocessing (parallel) ───────────────────────────────────

experiments: Experiments = Experiments(
    datasets=datasets,
    sizes=sizes,
    qualities=qualities,
    algos=["State"],
    path_to_data_dir=path_to_data_dir,
    signal_columns=signal_columns,
    seed=seeds[0],
)

experiments.prepare_state_data(max_workers=50)

# ── 2. Train / embed / MI (parallel) ────────────────────────────────────

for seed in seeds:

    experiments: Experiments = Experiments(
        datasets=datasets,
        sizes=sizes,
        qualities=qualities,
        algos=["State"],
        path_to_data_dir=path_to_data_dir,
        signal_columns=signal_columns,
        device=0,
        seed=seed,
    )

    experiments.parallel_run(
        sleep_time=0.2,
        retrain=True,
        reembed=True,
        recompute_mutual_information=True,
        early_stopping_patience=5,
        jobs_per_gpu=2,
        log_dir=f"{path_to_data_dir}/merfish/logs/state_run_seed_{seed}",
    )
