"""Run missing MI jobs from a CSV file in parallel.

Reads a CSV with columns (dataset, size, quality, algorithm, signal, seed),
groups by (dataset, size, quality, algorithm, seed), and runs each group
as a single_job via the scaling_laws Experiments API.

Usage:
    python run_missing_mi.py --max-workers 4
    python run_missing_mi.py --max-workers 8 --base-dir /mnt/nvme/noise_laws/data/
    python run_missing_mi.py --max-workers 4 --retrain --reembed
    python run_missing_mi.py --csv /path/to/other.csv --max-workers 2 --device 1
"""

import argparse
import csv
import multiprocessing
import random
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

# Dataset -> signal_columns mapping (from the existing run_*_whole.py scripts)
DATASET_SIGNAL_COLUMNS = {
    "PBMC": ["celltype.l3", "protein_counts"],
    "larry": ["index", "clone", "time"],
    "merfish": ["cur_idx", "ng_idx"],
    "shendure": ["author_day"],
}


def run_single_job(
    dataset: str,
    size: int,
    quality: float,
    algo: str,
    seed: int,
    base_dir: str,
    signal_columns: list[str],
    device: int,
    retrain: bool,
    reembed: bool,
    recompute_mi: bool,
) -> dict:
    """Run a single job in a subprocess. Returns a result dict."""
    job_key = f"{dataset}/{size}/{quality}/{algo}/seed={seed}"
    try:
        from scaling_laws.prepare.data import Experiments

        experiments = Experiments(
            datasets=[dataset],
            sizes=[size],
            qualities=[quality],
            algos=[algo],
            path_to_data_dir=base_dir,
            signal_columns=signal_columns,
            device=device,
            seed=seed,
        )
        experiments.single_job(
            dataset=dataset,
            size=size,
            quality=quality,
            algo=algo,
            device=device,
            retrain=retrain,
            reembed=reembed,
            recompute_mutual_information=recompute_mi,
        )
        return {"job": job_key, "status": "success"}
    except Exception as e:
        return {"job": job_key, "status": "failed", "error": f"{e}\n{traceback.format_exc()}"}


def main():
    parser = argparse.ArgumentParser(description="Run missing MI jobs from CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).parent / "missing_mi_on_disk.csv"),
        help="Path to CSV file with missing jobs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        required=True,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/igor/igor_repos/scaling_laws/data_local/",
        help="Base data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU device IDs, comma-separated (e.g. '0,1,2,3') or single ID. Defaults to all available GPUs.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Retrain models (default: False)",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        default=False,
        help="Re-embed data (default: False)",
    )
    parser.add_argument(
        "--no-recompute-mi",
        action="store_true",
        default=False,
        help="Skip MI recomputation (default: MI is recomputed)",
    )
    args = parser.parse_args()

    # Read CSV and group by (dataset, size, quality, algorithm, seed)
    jobs = defaultdict(set)
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["dataset"],
                int(row["size"]),
                float(row["quality"]),
                row["algorithm"],
                int(row["seed"]),
            )
            jobs[key].add(row["signal"])

    # Shuffle job order so parallel workers don't all hit the same dataset/size
    job_list = list(jobs.items())
    random.shuffle(job_list)
    print(f"Loaded {len(job_list)} unique jobs from {args.csv} (shuffled)")

    if args.device is not None:
        devices = [int(d) for d in args.device.split(",")]
    else:
        # Detect GPUs without importing torch (avoids CUDA init in main process)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            )
            devices = [int(line.strip()) for line in result.stdout.strip().splitlines()]
        except (FileNotFoundError, subprocess.CalledProcessError):
            devices = [0]
    print(f"Using {len(devices)} GPU(s): {devices}")

    recompute_mi = not args.no_recompute_mi
    results = {"success": 0, "failed": 0}
    failed_jobs = []

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as executor:
        futures = {}
        for i, ((dataset, size, quality, algo, seed), signals) in enumerate(job_list):
            device = devices[i % len(devices)]
            signal_columns = DATASET_SIGNAL_COLUMNS.get(dataset, list(signals))
            future = executor.submit(
                run_single_job,
                dataset=dataset,
                size=size,
                quality=quality,
                algo=algo,
                seed=seed,
                base_dir=args.base_dir,
                signal_columns=signal_columns,
                device=device,
                retrain=args.retrain,
                reembed=args.reembed,
                recompute_mi=recompute_mi,
            )
            futures[future] = (dataset, size, quality, algo, seed)

        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
            if result["status"] == "failed":
                failed_jobs.append(result)
                print(f"FAILED: {result['job']}: {result['error']}")
            else:
                print(f"OK: {result['job']}")

    print(f"\nDone: {results['success']} succeeded, {results['failed']} failed")
    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            print(f"  {job['job']}: {job['error'].splitlines()[0]}")


if __name__ == "__main__":
    main()
