"""Run STATE on merfish only (train + embed + MI).

Copy of ``2026-04-16_14-49_compute_state_all_datasets.py`` restricted to the
merfish dataset.  Assumes raw data and STATE preprocessing are already done.

Training protocol (set by scaling_laws.algo.state.State + Experiments):
  * validation every 500 optimizer steps (clamped to batches_per_epoch for
    tiny datasets so Lightning still validates at least once per epoch)
  * early stopping on ``validation/val_loss`` with patience=5 (triggered off
    the 500-step val checks, not off epoch boundaries)
  * fixed step budget of 15,000 optimizer steps for every (size, quality);
    no max_epochs cap — Lightning's epoch count is derived from max_steps so
    it stops as soon as 15k optimizer steps land. Early stopping (patience=5)
    can stop earlier if val_loss plateaus.
  * after training, the trainer writes three scalar files next to the model:
    ``train_loss.txt``, ``val_loss.txt``, ``test_loss.txt`` (test is a proxy
    for best val_loss, matching ``analysis/2026-04-15_14-43_compute_loss_scaling.py``)
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
    "merfish": {
        "sizes": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        "qualities": [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267,
                      0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        "signal_columns": ["cur_idx", "ng_idx"],
        "seeds": [1404, 2303, 2701],
        "jobs_per_gpu": 2,
    },
}

for ds_name, cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  {len(cfg['sizes'])} sizes x {len(cfg['qualities'])} qualities "
          f"x {len(cfg['seeds'])} seeds = {len(cfg['sizes']) * len(cfg['qualities']) * len(cfg['seeds'])} runs")
    print(f"  jobs_per_gpu={cfg['jobs_per_gpu']}")
    print(f"  Validation every 500 steps, early stopping patience=5, max_steps=15000 (no max_epochs cap), saving train/val/test_loss.txt")
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
            log_dir=f"{path_to_data_dir}/{ds_name}/logs/state_run_seed_{seed}",
        )

    print(f"\n[{ds_name}] Done.")

print("\n" + "=" * 70)
print("  MERFISH RUN COMPLETE")
print("=" * 70)
