"""Run STATE only for missing experiments across all datasets.

Scans for MI result files to identify which (dataset, size, quality) combos
are incomplete, then runs only those.  Order: larry, merfish, PBMC, shendure.

jobs_per_gpu=2 for larry/merfish/PBMC, jobs_per_gpu=1 for shendure.
"""

import sys
import os
from pathlib import Path
from itertools import product

# ── Auto-log ────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
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

# ── Imports ──────────────────────────────────────────────────────────────
from scaling_laws.prepare.data import Experiments

from scaling_laws.paths import DATA_DIR
path_to_data_dir = str(DATA_DIR)
DATA_ROOT = Path(path_to_data_dir)

# ── Dataset definitions (order: larry, merfish, PBMC, shendure) ──────────
DATASETS = [
    ("larry", {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311, 0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
        "signal_columns": ["clone"],
        "seed": 42,
        "jobs_per_gpu": 2,
    }),
    ("merfish", {
        "sizes": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        "qualities": [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        "signal_columns": ["cur_idx", "ng_idx"],
        "seed": 42,
        "jobs_per_gpu": 2,
    }),
    ("PBMC", {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422, 0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
        "signal_columns": ["celltype.l3", "protein_counts"],
        "seed": 42,
        "jobs_per_gpu": 2,
    }),
    ("shendure", {
        "sizes": [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
        "qualities": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        "signal_columns": ["author_day"],
        "seed": 42,
        "jobs_per_gpu": 1,
    }),
]


def find_missing(ds_name, cfg):
    """Return list of (size, quality) pairs missing any MI result."""
    seed = cfg["seed"]
    missing = []
    for sz, q in product(cfg["sizes"], cfg["qualities"]):
        # Check if ALL signal columns have MI files
        all_present = True
        for sig in cfg["signal_columns"]:
            mi_path = (
                DATA_ROOT / ds_name / str(sz) / str(q)
                / "results" / "State" / "model" / "MI" / str(seed)
                / f"Y_{sig}_{q}" / "lmi_mutual_information.txt"
            )
            if not mi_path.exists():
                all_present = False
                break
        if not all_present:
            missing.append((sz, q))
    return missing


# ── Main loop ────────────────────────────────────────────────────────────
for ds_name, cfg in DATASETS:
    seed = cfg["seed"]
    missing = find_missing(ds_name, cfg)
    total = len(cfg["sizes"]) * len(cfg["qualities"])

    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  Total grid: {total} | Already done: {total - len(missing)} | Missing: {len(missing)}")
    print(f"  jobs_per_gpu={cfg['jobs_per_gpu']}")
    print(f"{'='*70}")

    if not missing:
        print(f"  [SKIP] All {total} experiments complete for {ds_name}.\n")
        continue

    # Extract unique sizes and qualities from missing pairs
    missing_sizes = sorted(set(sz for sz, _ in missing))
    missing_qualities = sorted(set(q for _, q in missing))

    print(f"  Missing sizes:     {missing_sizes}")
    print(f"  Missing qualities: {missing_qualities}")
    print(f"  Missing combos:    {len(missing)} (will run {len(missing_sizes)} x {len(missing_qualities)} = {len(missing_sizes)*len(missing_qualities)} grid)")
    print()

    # The parallel_run products sizes x qualities, so some already-done combos
    # in the grid will be re-run.  That's fine -- retrain will redo them quickly
    # since checkpoints exist, or we can let it run fresh.
    # To be precise, we pass only missing_sizes and missing_qualities.
    experiments = Experiments(
        datasets=[ds_name],
        sizes=missing_sizes,
        qualities=missing_qualities,
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
