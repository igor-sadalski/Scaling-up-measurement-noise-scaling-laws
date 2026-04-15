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

import pandas as pd

from scaling_laws.s3_retriever import DATASET_SIZES, DATASET_QUALITIES

DATA_DIR = Path("/home/igor/noise_scaling/data")
DATASETS = ["PBMC", "larry", "merfish", "shendure"]
ALGOS = ["Geneformer", "SCVI", "State"]
NUM_GPUS = 8


def worker_geneformer(gpu_id, jobs, result_queue):
    """Worker: pin to GPU, import once, process all assigned jobs."""
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
                per_device_eval_batch_size=100,
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
            done += 1
            del model, trainer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL GPU{gpu_id} Geneformer {ds}/{sz}/{q}: {e}", flush=True)
            fail += 1
            torch.cuda.empty_cache()

    result_queue.put(("Geneformer", gpu_id, done, fail))


def worker_scvi(gpu_id, jobs, result_queue):
    """Worker: pin to GPU, import once, process all assigned jobs."""
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
            done += 1
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL GPU{gpu_id} SCVI {ds}/{sz}/{q}: {e}", flush=True)
            fail += 1
            torch.cuda.empty_cache()

    result_queue.put(("SCVI", gpu_id, done, fail))


# ── Step 1: Scan ─────────────────────────────────────────────────────────
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
            has_model = (base / "loss" / "metrics.csv").exists()
        has_loss = (base / "test_loss.txt").exists()
        scan.append({"dataset": ds, "size": sz, "quality": q, "algorithm": algo,
                      "has_model": has_model, "has_loss": has_loss})

df_scan = pd.DataFrame(scan)
summary = df_scan.groupby(["dataset", "algorithm"]).agg(
    models=("has_model", "sum"), losses=("has_loss", "sum"), total=("has_model", "count"),
).reset_index()
summary["missing"] = summary["models"] - summary["losses"]
print(summary.to_string(index=False), flush=True)

missing = df_scan[(df_scan["has_model"]) & (~df_scan["has_loss"])]
print(f"\nTotal missing: {len(missing)}", flush=True)
if len(missing) == 0:
    print("Nothing to do!")
    sys.exit(0)

# ── Step 2: STATE (inline, instant) ──────────────────────────────────────
state_missing = missing[missing["algorithm"] == "State"]
if len(state_missing) > 0:
    print(f"\nSTATE: {len(state_missing)} missing", flush=True)
    for _, row in state_missing.iterrows():
        ds, sz, q = row["dataset"], row["size"], row["quality"]
        metrics_path = DATA_DIR / ds / str(sz) / str(q) / "results" / "State" / "model" / "loss" / "metrics.csv"
        loss_path = DATA_DIR / ds / str(sz) / str(q) / "results" / "State" / "model" / "test_loss.txt"
        try:
            df = pd.read_csv(metrics_path)
            val = df["validation/val_loss"].dropna()
            if not val.empty:
                loss_path.parent.mkdir(parents=True, exist_ok=True)
                loss_path.write_text(f"{float(val.min()):.6f}")
        except Exception as e:
            print(f"  FAIL State {ds}/{sz}/{q}: {e}", flush=True)

# ── Step 3: Geneformer — 8 GPU workers ───────────────────────────────────
gf_jobs = [(r["dataset"], r["size"], r["quality"])
           for _, r in missing[missing["algorithm"] == "Geneformer"].iterrows()]
if gf_jobs:
    print(f"\nGeneformer: {len(gf_jobs)} missing → {NUM_GPUS} GPU workers", flush=True)
    # Split jobs across GPUs
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, job in enumerate(gf_jobs):
        chunks[i % NUM_GPUS].append(job)

    result_q = Queue()
    procs = []
    for gpu_id in range(NUM_GPUS):
        if chunks[gpu_id]:
            p = Process(target=worker_geneformer, args=(gpu_id, chunks[gpu_id], result_q))
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

# ── Step 4: SCVI — 8 GPU workers ─────────────────────────────────────────
scvi_jobs = [(r["dataset"], r["size"], r["quality"])
             for _, r in missing[missing["algorithm"] == "SCVI"].iterrows()]
if scvi_jobs:
    print(f"\nSCVI: {len(scvi_jobs)} missing → {NUM_GPUS} GPU workers", flush=True)
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, job in enumerate(scvi_jobs):
        chunks[i % NUM_GPUS].append(job)

    result_q = Queue()
    procs = []
    for gpu_id in range(NUM_GPUS):
        if chunks[gpu_id]:
            p = Process(target=worker_scvi, args=(gpu_id, chunks[gpu_id], result_q))
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
