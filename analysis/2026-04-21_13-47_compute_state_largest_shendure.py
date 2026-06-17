"""Re-run STATE on the 3 largest shendure sizes across all 10 qualities,
with checkpoint-based resume support.

Background: the original launch of this script on 2026-04-21 crashed on 21 of
30 jobs with ``[Errno 28] No space left on device`` — 16 concurrent GPUs each
writing 700 MB ckpts every 1 000 steps filled the 6.8 T nvme. The disk has
since been freed (data/merfish, larry, wandb were backed up to S3 and
removed), and ``scaling_laws.algo.state.State.train()`` has been updated to:

  * auto-resume from ``last.ckpt`` when one exists (skipping the old
    checkpoint-dir wipe), and
  * write ckpts every 10 000 steps with ``save_last=true`` (~10× fewer files).

This script now inventories each (size, quality) and does three things:

  1. SKIP combos already complete (val_loss.txt + embeddings.csv + MI file).
  2. TRIM stale 1 k-step checkpoints from partial runs (keep only
     ``last.ckpt``) to reclaim ~800 GB before resume starts.
  3. LAUNCH one ``Experiments.parallel_run`` per size with only the pending
     qualities. Partial runs resume from ``last.ckpt`` automatically; fresh
     runs (crashed before first ckpt) start from step 0.

The ≥1-epoch guarantee and step-based early stopping are unchanged —
``max_steps=160_000`` caps total training, ``min_es_steps=batches_per_epoch``
prevents ES from firing before one full epoch (enforced in
``STATE/state/src/state/emb/train/callbacks.py``).
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
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scaling_laws.paths import DATA_DIR

path_to_data_dir = str(DATA_DIR)
REPO_ROOT = Path(path_to_data_dir).parent / "modeling" / "Scaling-up-measurement-noise-scaling-laws"
SINGLE_JOB = REPO_ROOT / "single_job.py"

DATASETS = {
    "shendure": {
        "sizes": [774263, 2782559, 10000000],
        "qualities": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        "signal_columns": ["author_day"],
        "seeds": [42],
        "jobs_per_gpu": 2,
    },
}


# ── Inventory helpers ─────────────────────────────────────────────────────
def _run_dir_for(ds_name: str, size: int, quality: float) -> Path:
    return Path(path_to_data_dir) / ds_name / str(size) / str(quality) / "results" / "State"


def _ckpt_run_dirs(results_dir: Path) -> list[Path]:
    parent = results_dir / "model" / "checkpoints"
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.glob("state_scaling_*") if p.is_dir())


_REQUIRED_CFG_KEYS = {"experiment", "embeddings", "dataset", "model", "optimizer", "loss"}


def _config_valid(ds_name: str, size: int, quality: float) -> bool:
    """state_config.yaml present, parseable, and has the profile-specific
    ``embeddings.<profile>`` / ``dataset.<profile>`` sections referenced by
    the CLI overrides. The 2026-04-21 disk-full crash truncated at least one
    of these yamls mid-write — runs with bad configs will fail instantly on
    Hydra's compose step, so skip them here rather than letting them crash
    the sweep."""
    import yaml
    cfg = (Path(path_to_data_dir) / ds_name / str(size) / str(quality)
           / "preprocessed" / "state_data" / "state_config.yaml")
    if not cfg.exists():
        return False
    try:
        data = yaml.safe_load(cfg.read_text())
    except Exception:
        return False
    if not isinstance(data, dict) or _REQUIRED_CFG_KEYS - set(data.keys()):
        return False
    profile = f"scaling_{ds_name}_{size}_{str(quality).replace('.', '_')}"
    if profile not in (data.get("embeddings") or {}):
        return False
    if profile not in (data.get("dataset") or {}):
        return False
    return True


def inventory(ds_name: str, size: int, quality: float) -> str:
    """Return 'complete' | 'partial' | 'fresh' | 'no_config'."""
    results = _run_dir_for(ds_name, size, quality)
    model = results / "model"
    val_loss = model / "val_loss.txt"
    embeddings_csv = model / "embeddings.csv"
    mi_any = list((model / "MI").glob("*/Y_author_day*/lmi_mutual_information.txt"))
    if val_loss.exists() and embeddings_csv.exists() and mi_any:
        return "complete"
    if not _config_valid(ds_name, size, quality):
        return "no_config"
    for run_dir in _ckpt_run_dirs(results):
        if any(run_dir.glob("*.ckpt")):
            return "partial"
    return "fresh"


def trim_legacy_ckpts(ds_name: str, size: int, quality: float) -> int:
    """Delete every ``*.ckpt`` except ``last.ckpt`` in the run dir.

    Returns bytes freed. Safe to call on complete or fresh runs — no-op if
    there's no ``last.ckpt`` to anchor on (we refuse to delete otherwise, so
    we never strand a partial run without a resume point).
    """
    results = _run_dir_for(ds_name, size, quality)
    freed = 0
    for run_dir in _ckpt_run_dirs(results):
        last = run_dir / "last.ckpt"
        if not last.exists():
            continue  # don't risk stranding a run without a resume anchor
        for p in run_dir.glob("*.ckpt"):
            if p.name == "last.ckpt":
                continue
            try:
                sz = p.stat().st_size
                p.unlink()
                freed += sz
            except FileNotFoundError:
                pass
    return freed


for ds_name, cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  {len(cfg['sizes'])} sizes x {len(cfg['qualities'])} qualities x "
          f"{len(cfg['seeds'])} seeds = {len(cfg['sizes']) * len(cfg['qualities']) * len(cfg['seeds'])} runs (nominal)")
    print(f"{'='*70}\n")

    # ── Phase 1: inventory ────────────────────────────────────────────────
    status: dict[int, dict[float, str]] = {s: {} for s in cfg["sizes"]}
    counts = {"complete": 0, "partial": 0, "fresh": 0, "no_config": 0}
    for size in cfg["sizes"]:
        for q in cfg["qualities"]:
            st = inventory(ds_name, size, q)
            status[size][q] = st
            counts[st] += 1

    print(f"[{ds_name}] Inventory: "
          f"{counts['complete']} complete, "
          f"{counts['partial']} partial (resume), "
          f"{counts['fresh']} fresh, "
          f"{counts['no_config']} missing/corrupt config (skipped)")
    print(f"{'size':>10}  {'quality':>10}  status")
    for size in cfg["sizes"]:
        for q in cfg["qualities"]:
            print(f"{size:>10}  {q:>10}  {status[size][q]}")

    # ── Phase 2: trim legacy 1k-step checkpoints on partial runs ─────────
    total_freed = 0
    for size in cfg["sizes"]:
        for q, st in status[size].items():
            if st != "partial":
                continue
            freed = trim_legacy_ckpts(ds_name, size, q)
            total_freed += freed
            if freed > 0:
                print(f"[trim] shendure/{size}/{q}: freed {freed/1024**3:.1f} GB (kept last.ckpt)")
    print(f"[trim] TOTAL freed: {total_freed/1024**3:.1f} GB")

    # ── Phase 3: launch all runnable jobs as a single flat pool ──────────
    # Runnable = anything that isn't already complete AND has a valid config.
    # Jobs across all 3 sizes go into one pool; a GPU-slot queue enforces
    # exactly `jobs_per_gpu` concurrent jobs per GPU across 8 GPUs, so all
    # 8*jobs_per_gpu = 16 slots stay saturated as long as work remains.
    # Submit largest sizes first: 10M tiers take longest, so starting them
    # early means the shorter 2.78M / 774k jobs backfill as slots free up.
    num_gpus = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().count("\n")) + 1
    jobs_per_gpu = cfg["jobs_per_gpu"]
    total_slots = num_gpus * jobs_per_gpu
    print(f"\n[{ds_name}] Scheduler: {num_gpus} GPUs x {jobs_per_gpu} jobs/GPU = {total_slots} concurrent slots")

    gpu_slots: "queue.Queue[int]" = queue.Queue()
    for gpu_idx in range(num_gpus):
        for _ in range(jobs_per_gpu):
            gpu_slots.put(gpu_idx)

    job_log_dir = Path(path_to_data_dir) / ds_name / "logs" / "state_run_largest_flat"
    job_log_dir.mkdir(parents=True, exist_ok=True)

    pending_jobs: list[tuple[int, float, int]] = []
    for seed in cfg["seeds"]:
        for size in sorted(cfg["sizes"], reverse=True):  # 10M first
            for q in cfg["qualities"]:
                if status[size][q] in ("partial", "fresh"):
                    pending_jobs.append((size, q, seed))

    print(f"[{ds_name}] Submitting {len(pending_jobs)} jobs to the pool:")
    for (s, q, sd) in pending_jobs:
        print(f"    {s:>10}  {q:>10}  seed={sd}  [{status[s][q]}]")

    print_lock = threading.Lock()

    def run_one_job(size: int, quality: float, seed: int) -> tuple[int, float, int, int]:
        """Claim a GPU slot, run single_job.py as a subprocess, release slot."""
        gpu = gpu_slots.get()
        log_path = job_log_dir / f"{size}_{quality}_seed{seed}.log"
        try:
            with print_lock:
                print(f"[start]  {size:>10} q={quality} seed={seed} -> GPU {gpu}  log={log_path.name}")
            cmd = [
                "python", str(SINGLE_JOB),
                "--sizes", str(size),
                "--qualities", str(quality),
                "--algos", "State",
                "--base_dir", path_to_data_dir,
                "--device", str(gpu),
                "--max_epochs", "1",
                "--early_stopping_patience", "5",
                "--dataset", ds_name,
                "--retrain", "true",
                "--reembed", "true",
                "--recompute_mutual_information", "true",
                "--recompute_loss", "false",
                "--seed", str(seed),
                "--max_steps", "160000",
                "--signal_columns", *cfg["signal_columns"],
            ]
            with open(log_path, "w") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            rc = proc.returncode
            with print_lock:
                tag = "done" if rc == 0 else "FAIL"
                print(f"[{tag}]   {size:>10} q={quality} seed={seed} (GPU {gpu}, rc={rc})")
            return (size, quality, seed, rc)
        finally:
            gpu_slots.put(gpu)

    succeeded = failed = 0
    with ThreadPoolExecutor(max_workers=total_slots) as pool:
        futures = [pool.submit(run_one_job, s, q, sd) for (s, q, sd) in pending_jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"[{ds_name}] jobs"):
            _, _, _, rc = fut.result()
            if rc == 0:
                succeeded += 1
            else:
                failed += 1

    print(f"\n[{ds_name}] Pool complete: {succeeded} succeeded, {failed} failed "
          f"(out of {len(pending_jobs)} submitted)")

print("\n" + "=" * 70)
print("  LARGEST SHENDURE SIZES COMPLETE (774263, 2782559, 10000000)")
print("=" * 70)
