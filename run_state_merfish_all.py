"""
Run STATE on ALL MERFISH size/quality combinations.
Training dynamics match Geneformer:
  - Same epoch formula: max(1, 10 * 10_000_000 // size)
  - Early stopping patience=5 on val_loss (every 1000 steps)
  - Same architecture: 256 hidden, 4 heads, 3 layers, 512 FFN
  - Same batch_size=64, lr=1e-3, weight_decay=0.001, dropout=0.02

Usage:
    conda activate modeling
    python run_state_merfish_all.py
"""

import random
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────

DATA_DIR = "/mnt/nvme/noise_laws/data"
DATASET = "merfish"
SIGNAL_COLUMNS = ["cur_idx", "ng_idx"]
SEEDS = [42]

SIZES = list(map(int, np.logspace(np.log10(60000), 2, 10)))
QUALITIES = list(map(lambda x: round(x, 7), np.logspace(0, np.log10(10 / 367), 10)))

MAX_WORKERS = 3
MEM_LIMIT_MB = 20_000


def get_max_epochs(size: int) -> int:
    """Same formula as Geneformer: max(1, 10 * 10M / size)."""
    return max(1, int(10 * (10_000_000 / size)))


# ── Worker ───────────────────────────────────────────────────────────────

def run_single(dataset, size, quality, seed, max_epochs, device):
    try:
        from scaling_laws.prepare.data import Experiments

        exp = Experiments(
            datasets=[dataset],
            sizes=[size],
            qualities=[quality],
            algos=["State"],
            path_to_data_dir=DATA_DIR,
            signal_columns=SIGNAL_COLUMNS,
            seed=seed,
        )

        base = Path(DATA_DIR) / dataset / str(size) / str(quality) / "results" / "State" / "model"
        emb_path = base / "embeddings.csv"
        mi_done = list((base / "MI").rglob("lmi_mutual_information.txt")) if (base / "MI").exists() else []

        if emb_path.exists() and len(mi_done) >= 2:
            return f"SKIP size={size} q={quality} seed={seed}"

        exp.single_job(
            dataset=dataset,
            size=size,
            quality=quality,
            algo="State",
            max_epochs=max_epochs,
            early_stopping_patience=5,
            device=device,
            retrain=not emb_path.exists(),
            reembed=not emb_path.exists(),
            recompute_mutual_information=True,
        )
        return f"OK size={size} q={quality} seed={seed} GPU={device}"
    except Exception as e:
        return f"FAIL size={size} q={quality} seed={seed}: {e}\n{traceback.format_exc()}"


def get_free_gpus(mem_limit_mb=MEM_LIMIT_MB):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
        free = []
        for line in out.strip().split("\n"):
            gpu_id, mem_used = map(int, line.split(", "))
            if mem_used < mem_limit_mb:
                free.append(gpu_id)
        return free
    except Exception:
        return list(range(8))


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jobs = []
    for seed in SEEDS:
        for size in SIZES:
            for quality in QUALITIES:
                jobs.append((DATASET, size, quality, seed, get_max_epochs(size)))

    random.shuffle(jobs)
    print(f"Total jobs: {len(jobs)}")
    for s in SIZES:
        print(f"  size={s:>6d}: max_epochs={get_max_epochs(s):>10,} (early stopping patience=5)")
    print()

    completed, failed, skipped = 0, 0, 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        active = {}
        job_iter = iter(jobs)

        for _ in range(MAX_WORKERS):
            try:
                dataset, size, quality, seed, max_epochs = next(job_iter)
            except StopIteration:
                break
            free = get_free_gpus()
            if not free:
                time.sleep(30)
                free = get_free_gpus()
            device = free[len(active) % len(free)] if free else 0
            future = executor.submit(run_single, dataset, size, quality, seed, max_epochs, device)
            active[future] = (size, quality, seed, device)
            print(f"Submitted: size={size} q={quality} epochs={max_epochs:,} GPU={device}")
            time.sleep(0.5)

        while active:
            done_future = next(as_completed(active))
            info = active.pop(done_future)
            try:
                result = done_future.result()
                if result.startswith("SKIP"):
                    skipped += 1
                elif result.startswith("OK"):
                    completed += 1
                else:
                    failed += 1
                elapsed = time.time() - t_start
                total_done = completed + failed + skipped
                print(f"[{total_done}/{len(jobs)}] ({elapsed/60:.0f}m) {result[:150]}")
            except Exception as e:
                failed += 1
                print(f"[ERROR] {info}: {e}")

            try:
                dataset, size, quality, seed, max_epochs = next(job_iter)
                free = get_free_gpus()
                device = free[len(active) % len(free)] if free else 0
                future = executor.submit(run_single, dataset, size, quality, seed, max_epochs, device)
                active[future] = (size, quality, seed, device)
                time.sleep(0.5)
            except StopIteration:
                pass

    elapsed = time.time() - t_start
    print(f"\nALL DONE in {elapsed/60:.1f} min — OK={completed}, Skip={skipped}, Fail={failed}")

    # ── Collect results ──────────────────────────────────────────────────
    import pandas as pd

    ALL_ALGOS = ["Geneformer", "SCVI", "PCA", "RandomProjection", "State"]
    rows = []
    for size in SIZES:
        for quality in QUALITIES:
            base = Path(DATA_DIR) / DATASET / str(size) / str(quality) / "results"
            for algo in ALL_ALGOS:
                mi_base = base / algo / "model" / "MI"
                if not mi_base.exists():
                    continue
                for seed_dir in sorted(mi_base.iterdir()):
                    for sig_dir in sorted(seed_dir.iterdir()):
                        mi_file = sig_dir / "lmi_mutual_information.txt"
                        if mi_file.exists():
                            rows.append({
                                "size": size, "quality": quality, "algo": algo,
                                "seed": int(seed_dir.name), "signal": sig_dir.name,
                                "MI": float(mi_file.read_text().strip()),
                            })

    if rows:
        df = pd.DataFrame(rows)
        out = Path(DATA_DIR) / DATASET / "all_state_mi_results.csv"
        df.to_csv(out, index=False)
        print(f"\nResults ({len(df)} rows) saved to: {out}")
