"""Run STATE on MERFISH (all sizes x all qualities).

Assumes data is already prepared (sampled, downsampled, tokenized)
by run_merfish_whole.py. Only runs STATE preprocessing + training.

Hyperparameters from notebook 2026-04-13_20-36:
  - max_epochs=10, early_stopping_patience=5
  - pad_length=2048, emsize=256, 4 heads, 3 layers, 512 FFN
  - max_lr=1e-4, dropout=0.1, batch_size=64
  - LMI max_epochs=300
"""

import shutil
from pathlib import Path
from scaling_laws.prepare.data import Experiments
import numpy as np

datasets = ["merfish"]
sizes = list(map(int, np.logspace(np.log10(60000), 2, 10)))
qualities = list(map(lambda x: round(x, 7), np.logspace(0, np.log10(10 / 367), 10)))
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

# ── 3. Train / embed / MI (parallel) ────────────────────────────────────

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
        max_workers=8,
        sleep_time=0.2,
        retrain=True,
        reembed=True,
        recompute_mutual_information=True,
        max_epochs=10,
        early_stopping_patience=5,
        jobs_per_gpu=2,
        log_dir=f"{path_to_data_dir}/merfish/logs/state_run_seed_{seed}",
    )
