"""Compute test loss for all existing Geneformer, SCVI, and State experiments.

8 worker processes, each pinned to a GPU. Each worker imports once, then
processes its share of jobs sequentially — avoids both subprocess overhead
and single-GPU bottleneck.
"""
import os
import sys
from pathlib import Path
from itertools import product
from multiprocessing import Process, Queue

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
    """Write to both a file and the original stream.

    Safe for multiprocessing: write/flush silently ignore broken-pipe errors,
    and close() is a no-op so forked children don't close the shared FD.
    """
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        try:
            self.stream.write(data)
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def flush(self):
        try:
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.log_file.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        pass  # Don't close underlying streams from forked children


_log_fh = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)

import scaling_laws  # noqa: F401 — activates timestamped print
import pandas as pd

from scaling_laws.s3_retriever import DATASET_SIZES, DATASET_QUALITIES

print(f"Logging to {LOG_PATH}")

from scaling_laws.paths import DATA_DIR
DATASETS = ["PBMC", "larry", "merfish", "shendure"]
ALGOS = ["Geneformer", "SCVI", "State"]
NUM_GPUS = 8
JOBS_PER_GPU = 1
NUM_WORKERS = NUM_GPUS * JOBS_PER_GPU


def _geneformer_batch_size(dataset: str) -> int:
    """Pick eval batch size per dataset to avoid OOM on 40 GB GPUs."""
    return {"shendure": 8, "larry": 32}.get(dataset, 64)


def worker_geneformer(gpu_id, jobs, result_queue):
    """Worker: pin to GPU, import once, process all assigned jobs."""
    # Reset stdout/stderr to avoid inherited Tee broken-pipe cascade
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import pickle
    from transformers import BertForMaskedLM, TrainingArguments
    from datasets import load_from_disk
    from geneformer import GeneformerPretrainer

    done, fail = 0, 0
    for ds, sz, q in jobs:
        base_dir = DATA_DIR / ds / str(sz) / str(q)
        model_path = base_dir / "results" / "Geneformer" / "model"
        loss_path = model_path / "test_loss.txt"
        test_path = base_dir.parent.parent / "test" / str(q) / "preprocessed" / "tokenized.dataset"
        token_dict_path = DATA_DIR / ds / "utils" / "token_dict.pkl"
        lengths_path = base_dir / "preprocessed" / "lengths.pkl"

        try:
            with open(token_dict_path, "rb") as fp:
                token_dictionary = pickle.load(fp)
            test_dataset = load_from_disk(str(test_path))
            model = BertForMaskedLM.from_pretrained(str(model_path)).to("cuda:0")

            eval_args = TrainingArguments(
                output_dir="/tmp/gf_eval_tmp",
                per_device_eval_batch_size=_geneformer_batch_size(ds),
                do_eval=True,
                report_to="none",
            )
            trainer = GeneformerPretrainer(
                model=model, args=eval_args, eval_dataset=test_dataset,
                token_dictionary=token_dictionary,
                example_lengths_file=str(lengths_path),
            )
            metrics = trainer.evaluate()
            loss_path.write_text(f"{metrics['eval_loss']:.6f}")
            print(f"  OK   GPU{gpu_id} Geneformer [{ds},{q},{sz}]: loss={metrics['eval_loss']:.6f}", flush=True)
            done += 1
            del model, trainer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL GPU{gpu_id} Geneformer [{ds},{q},{sz}]: {e}", flush=True)
            fail += 1
            torch.cuda.empty_cache()

    result_queue.put(("Geneformer", gpu_id, done, fail))


def worker_scvi(gpu_id, jobs, result_queue):
    """Worker: pin to GPU, import once, process all assigned jobs."""
    # Reset stdout/stderr to avoid inherited Tee broken-pipe cascade
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import scvi as scvi_lib
    import anndata as ad

    done, fail = 0, 0
    for ds, sz, q in jobs:
        base_dir = DATA_DIR / ds / str(sz) / str(q)
        model_path = base_dir / "results" / "SCVI" / "model"
        loss_path = model_path / "test_loss.txt"
        test_h5ad = base_dir.parent.parent / "test" / str(q) / "preprocessed" / "preprocessed.h5ad"

        try:
            adata_test = ad.read_h5ad(test_h5ad, backed="r")
            vae = scvi_lib.model.SCVI.load(dir_path=model_path, adata=adata_test)
            elbo = vae.get_elbo(adata_test)
            loss_path.write_text(f"{elbo:.6f}")
            print(f"  OK   GPU{gpu_id} SCVI [{ds},{q},{sz}]: elbo={elbo:.6f}", flush=True)
            done += 1
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL GPU{gpu_id} SCVI [{ds},{q},{sz}]: {e}", flush=True)
            fail += 1
            torch.cuda.empty_cache()

    result_queue.put(("SCVI", gpu_id, done, fail))


def find_state_metrics_csv(model_dir: Path):
    """Return the path to STATE's metrics.csv, or None if not found.

    Two possible layouts:
      1. model/loss/metrics.csv            (newer runs)
      2. model/checkpoints/<run>/version_0/metrics.csv  (PyTorch Lightning default)

    Only returns a path if the CSV actually has loss data (not just
    step + learning_rate from a crashed early run).
    """
    candidates = []
    p = model_dir / "loss" / "metrics.csv"
    if p.exists():
        candidates.append(p)
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(ckpt_dir.glob("*/version_0/metrics.csv"))

    for csv_path in candidates:
        try:
            header = csv_path.read_text().split("\n", 1)[0]
            if "val_loss" in header or "train_loss" in header:
                return csv_path
        except Exception:
            continue
    return None


def read_state_loss(metrics_path: Path) -> float | None:
    """Read best loss from a STATE metrics CSV.

    Tries validation/val_loss first, falls back to trainer/train_loss.
    Returns None if no valid loss found.
    """
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    for col in ("validation/val_loss", "trainer/train_loss"):
        if col in df.columns:
            vals = df[col].dropna()
            if not vals.empty:
                return float(vals.min())
    return None


# ── Step 1: Scan & completeness table ────────────────────────────────────
print("Scanning ...", flush=True)
scan = []
for ds in DATASETS:
    for sz, q, algo in product(DATASET_SIZES[ds], DATASET_QUALITIES[ds], ALGOS):
        base = DATA_DIR / ds / str(sz) / str(q) / "results" / algo / "model"
        has_model = False
        if algo == "Geneformer":
            has_model = (base / "model.safetensors").exists()
        elif algo == "SCVI":
            has_model = (base / "model.pt").exists()
        elif algo == "State":
            has_model = find_state_metrics_csv(base) is not None
        has_loss = (base / "test_loss.txt").exists()
        scan.append({"dataset": ds, "size": sz, "quality": q, "algorithm": algo,
                      "has_model": has_model, "has_loss": has_loss})

df_scan = pd.DataFrame(scan)

# Completeness table: found / expected / missing per dataset x algorithm
pivot = df_scan.groupby(["dataset", "algorithm"]).agg(
    has_model=("has_model", "sum"),
    has_loss=("has_loss", "sum"),
    expected=("has_model", "count"),
).reset_index()
pivot["missing_loss"] = pivot["has_model"] - pivot["has_loss"]

print("\n── Completeness (has_model / has_loss / expected / to_compute) ──", flush=True)
for ds in DATASETS:
    ds_rows = pivot[pivot["dataset"] == ds]
    parts = []
    for _, r in ds_rows.iterrows():
        parts.append(f"  {r['algorithm']:>12s}: model={int(r['has_model']):3d}  "
                     f"loss={int(r['has_loss']):3d}  "
                     f"expected={int(r['expected']):3d}  "
                     f"to_compute={int(r['missing_loss']):3d}")
    print(f"\n{ds}:", flush=True)
    for p in parts:
        print(p, flush=True)

missing = df_scan[(df_scan["has_model"]) & (~df_scan["has_loss"])]
print(f"\nTotal jobs to compute: {len(missing)}", flush=True)
if len(missing) == 0:
    print("Nothing to do!")
    sys.exit(0)

# ── Step 2: STATE (inline, instant) ──────────────────────────────────────
state_missing = missing[missing["algorithm"] == "State"]
if len(state_missing) > 0:
    print(f"\nSTATE: {len(state_missing)} missing", flush=True)
    ok, fail = 0, 0
    for _, row in state_missing.iterrows():
        ds, sz, q = row["dataset"], row["size"], row["quality"]
        model_dir = DATA_DIR / ds / str(sz) / str(q) / "results" / "State" / "model"
        metrics_path = find_state_metrics_csv(model_dir)
        loss_path = model_dir / "test_loss.txt"
        if metrics_path is None:
            print(f"  SKIP State [{ds},{q},{sz}]: no valid metrics CSV", flush=True)
            fail += 1
            continue
        loss_val = read_state_loss(metrics_path)
        if loss_val is not None:
            loss_path.parent.mkdir(parents=True, exist_ok=True)
            loss_path.write_text(f"{loss_val:.6f}")
            print(f"  OK   State [{ds},{q},{sz}]: loss={loss_val:.6f}", flush=True)
            ok += 1
        else:
            print(f"  FAIL State [{ds},{q},{sz}]: no loss values in {metrics_path}", flush=True)
            fail += 1
    print(f"State: {ok} succeeded, {fail} failed", flush=True)

# ── Step 3: Geneformer — GPU workers (JOBS_PER_GPU per GPU) ─────────────
gf_jobs = [(r["dataset"], r["size"], r["quality"])
           for _, r in missing[missing["algorithm"] == "Geneformer"].iterrows()]
if gf_jobs:
    print(f"\nGeneformer: {len(gf_jobs)} missing → {NUM_WORKERS} workers ({JOBS_PER_GPU}/gpu)", flush=True)
    chunks = [[] for _ in range(NUM_WORKERS)]
    for i, job in enumerate(gf_jobs):
        chunks[i % NUM_WORKERS].append(job)

    result_q = Queue()
    procs = []
    for worker_id in range(NUM_WORKERS):
        if chunks[worker_id]:
            gpu_id = worker_id % NUM_GPUS
            p = Process(target=worker_geneformer, args=(gpu_id, chunks[worker_id], result_q))
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    total_done, total_fail = 0, 0
    while not result_q.empty():
        algo, gpu, d, f = result_q.get()
        total_done += d
        total_fail += f
    print(f"Geneformer: {total_done} succeeded, {total_fail} failed", flush=True)

# ── Step 4: SCVI — GPU workers (JOBS_PER_GPU per GPU) ────────────────────
scvi_jobs = [(r["dataset"], r["size"], r["quality"])
             for _, r in missing[missing["algorithm"] == "SCVI"].iterrows()]
if scvi_jobs:
    print(f"\nSCVI: {len(scvi_jobs)} missing → {NUM_WORKERS} workers ({JOBS_PER_GPU}/gpu)", flush=True)
    chunks = [[] for _ in range(NUM_WORKERS)]
    for i, job in enumerate(scvi_jobs):
        chunks[i % NUM_WORKERS].append(job)

    result_q = Queue()
    procs = []
    for worker_id in range(NUM_WORKERS):
        if chunks[worker_id]:
            gpu_id = worker_id % NUM_GPUS
            p = Process(target=worker_scvi, args=(gpu_id, chunks[worker_id], result_q))
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    total_done, total_fail = 0, 0
    while not result_q.empty():
        algo, gpu, d, f = result_q.get()
        total_done += d
        total_fail += f
    print(f"SCVI: {total_done} succeeded, {total_fail} failed", flush=True)

print("\nAll done!", flush=True)
