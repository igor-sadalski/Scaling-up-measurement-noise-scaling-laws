"""Run STATE on merfish: size scaling at q=1.0, then quality scaling at max size.

1. All 10 sizes at quality=1.0
2. All 10 qualities at size=60,000
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

ALL_SIZES = [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000]
ALL_QUALITIES = [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0]

RUNS = [
    {
        "label": "merfish size scaling (quality=1.0)",
        "sizes": ALL_SIZES,
        "qualities": [1.0],
    },
    {
        "label": "merfish quality scaling (size=60000)",
        "sizes": [60000],
        "qualities": ALL_QUALITIES,
    },
]

for run in RUNS:
    print(f"\n{'='*70}")
    print(f"  {run['label']}")
    print(f"  {len(run['sizes'])} sizes x {len(run['qualities'])} qualities")
    print(f"{'='*70}\n")

    for seed in [42]:
        print(f"\n  Training + MI with seed={seed}...")
        experiments = Experiments(
            datasets=["merfish"],
            sizes=run["sizes"],
            qualities=run["qualities"],
            algos=["State"],
            path_to_data_dir=path_to_data_dir,
            signal_columns=["cur_idx", "ng_idx"],
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

    print(f"\n  Done: {run['label']}")

print("\n" + "=" * 70)
print("  MERFISH SIZE + QUALITY SCALING COMPLETE")
print("=" * 70)
