"""Preprocess STATE data for all datasets.

Runs only the STATE preprocessing step (state emb preprocess) for every
dataset/size/quality combination. No training, embedding, or MI.
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

DATASETS = {
    "merfish": {
        "sizes": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        "qualities": [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        "signal_columns": ["cur_idx", "ng_idx"],
        "seeds": [1404, 2303, 2701],
        "prep_workers": 50,
    },
    "PBMC": {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422, 0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
        "signal_columns": ["celltype.l3", "protein_counts"],
        "seeds": [42, 2303, 2701],
        "prep_workers": 50,
    },
    "larry": {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311, 0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
        "signal_columns": ["clone"],
        "seeds": [42, 1404, 2701],
        "prep_workers": 50,
    },
    "shendure": {
        "sizes": [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
        "qualities": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        "signal_columns": ["author_day"],
        "seeds": [42],
        "prep_workers": 16,
    },
}

for ds_name, cfg in DATASETS.items():
    n_combos = len(cfg["sizes"]) * len(cfg["qualities"])
    print(f"\n{'='*70}")
    print(f"  PREPROCESS: {ds_name} ({n_combos} size x quality combos, {cfg['prep_workers']} workers)")
    print(f"{'='*70}\n")

    experiments = Experiments(
        datasets=[ds_name],
        sizes=cfg["sizes"],
        qualities=cfg["qualities"],
        algos=["State"],
        path_to_data_dir=path_to_data_dir,
        signal_columns=cfg["signal_columns"],
        seed=cfg["seeds"][0],
    )
    experiments.prepare_state_data(max_workers=cfg["prep_workers"])

    print(f"[{ds_name}] Preprocessing done.")

print("\n" + "=" * 70)
print("  ALL PREPROCESSING COMPLETE")
print("=" * 70)
