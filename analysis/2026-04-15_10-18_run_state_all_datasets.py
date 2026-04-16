"""Run STATE on all datasets (larry, merfish, shendure).

PBMC is commented out — already has sufficient runs.

Runs a single seed per dataset to fill the grid quickly.
Epoch scaling: max_epochs = max(1, 10 * max_dataset_size / size),
so the largest size per dataset trains for 10 epochs and smaller
sizes train proportionally longer (with early stopping patience=5).

Order: larry, shendure, then merfish.
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

path_to_data_dir = "/home/igor/noise_scaling/data"

DATASETS = {
    # "PBMC": {
    #     "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
    #     "qualities": [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422, 0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
    #     "signal_columns": ["celltype.l3", "protein_counts"],
    #     "seed": 42,
    #     "jobs_per_gpu": 2,
    # },
    "larry": {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311, 0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
        "signal_columns": ["clone"],
        "seed": 42,
        "jobs_per_gpu": 2,
    },
    "shendure": {
        "sizes": [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
        "qualities": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        "signal_columns": ["author_day"],
        "seed": 42,
        "jobs_per_gpu": 1,
    },
    "merfish": {
        "sizes": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        "qualities": [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        "signal_columns": ["cur_idx", "ng_idx"],
        "seed": 42,
        "jobs_per_gpu": 2,
    },
}

for ds_name, cfg in DATASETS.items():
    seed = cfg["seed"]
    n_runs = len(cfg["sizes"]) * len(cfg["qualities"])
    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  {len(cfg['sizes'])} sizes x {len(cfg['qualities'])} qualities "
          f"x 1 seed ({seed}) = {n_runs} runs")
    print(f"  jobs_per_gpu={cfg['jobs_per_gpu']}")
    print(f"{'='*70}\n")

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
        early_stopping_patience=5,
        jobs_per_gpu=cfg["jobs_per_gpu"],
        log_dir=f"{path_to_data_dir}/{ds_name}/logs/state_run_seed_{seed}",
    )

    print(f"\n[{ds_name}] Done.")

print("\n" + "=" * 70)
print("  ALL DATASETS COMPLETE")
print("=" * 70)
