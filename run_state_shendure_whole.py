"""Run STATE on shendure (10 sizes x 10 qualities, single seed).

Assumes data is already prepared (sampled, downsampled, tokenized) by
run_shendure_whole.py -- this script only runs STATE preprocessing + training.

Sizes/qualities mirror analysis/2026-04-15_10-18_run_state_all_datasets.py.
jobs_per_gpu=1 because shendure includes very large sizes (up to 10M cells)
that cannot fit two STATE jobs per GPU.
"""

import sys
from pathlib import Path

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

from scaling_laws.prepare.data import Experiments
from scaling_laws.paths import DATA_DIR

datasets = ["shendure"]
sizes = [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000]
qualities = [
    0.004,
    0.0073875,
    0.0136438,
    0.0251984,
    0.0465384,
    0.0859506,
    0.1587401,
    0.2931733,
    0.5414548,
    1.0,
]
signal_columns = ["author_day"]
seeds = [42]
path_to_data_dir = str(DATA_DIR)

# 1. STATE preprocessing (parallel)
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

# 2. Train / embed / MI
for seed in seeds:
    experiments = Experiments(
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
        jobs_per_gpu=1,
        log_dir=f"{path_to_data_dir}/shendure/logs/state_run_seed_{seed}",
    )
