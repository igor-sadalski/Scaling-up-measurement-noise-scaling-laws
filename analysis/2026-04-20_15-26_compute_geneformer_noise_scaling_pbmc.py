"""Geneformer noise-scaling sweep on PBMC.

Fine-tunes the **pretrained** ctheodoris/Geneformer (gc104M, 316M params,
1152H/18L/18-head) on each cell of the 10 sizes x 10 qualities PBMC grid.
For each (size, quality):

  1. Re-tokenize train h5ad and the per-quality test h5ad with the official
     gc104M token dictionary (idempotent — skip if already done).
  2. Continue MLM fine-tuning the pretrained checkpoint for EPOCHS epochs.
  3. Extract cell embeddings on the test set with EmbExtractor.
  4. Estimate MI between embeddings and the protein_counts signal via latentmi.

Outputs go to /home/igor/noise_scaling/data/other/geneformer_noise_scaling/,
mirroring data/other/hp_tunning/. Existing PBMC/Geneformer results in
/home/igor/noise_scaling/data/PBMC/.../results/Geneformer/ are not touched.

Parallelism: 1 job per GPU (the 316M Geneformer + activations push memory).
"""

from __future__ import annotations

import atexit
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# -- Auto-log: tee stdout/stderr to .log file next to this script --------
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
atexit.register(_log_f.close)
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
print(f"Logging to {LOG_PATH}")


# -- Configuration -------------------------------------------------------
DATASET = "PBMC"
SIZES = [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000]
QUALITIES = [
    0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422,
    0.050973, 0.1072766, 0.225772, 0.4751547, 1.0,
]
SEEDS = [42]
EPOCHS = 3                  # short fine-tune (small datasets); capped by MAX_STEPS
MAX_STEPS = 1500            # hard cap on optimizer steps so 100k-cell jobs finish in <2h
PER_DEVICE_TRAIN_BS = 1     # 316M model + 2048-token seqs; target ~10GB VRAM
GRAD_ACCUM = 16             # effective batch = 16
PER_DEVICE_EVAL_BS = 2      # inference: small BS to stay under 10GB VRAM
LR = 5e-5                   # typical fine-tune lr (vs 1e-3 for from-scratch)
WARMUP_STEPS = 100
MODEL_INPUT_SIZE = 2048     # tokenizer truncation length
EMB_MAX_NCELLS = 5000       # cap test cells used for embedding+MI (keeps embed step bounded)

DATA_DIR = Path("/home/igor/noise_scaling/data")
OUTPUT_DIR = DATA_DIR / "other" / "geneformer_noise_scaling"
GENEFORMER_DIR = Path(
    "/home/igor/noise_scaling/modeling/Scaling-up-measurement-noise-scaling-laws/Geneformer"
)
GENE_DICT_DIR = GENEFORMER_DIR / "geneformer"
TOKEN_DICT_PATH = GENE_DICT_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN_PATH = GENE_DICT_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_NAME_ID_PATH = GENE_DICT_DIR / "gene_name_id_dict_gc104M.pkl"
ENSEMBL_MAP_PATH = GENE_DICT_DIR / "ensembl_mapping_dict_gc104M.pkl"

JOBS_PER_GPU = 1


def _set_seed(seed: int) -> None:
    import torch  # local import — workers only
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -- Re-tokenize PBMC h5ad with the official gc104M dictionary -----------

def _retokenize_h5ad(src_h5ad: Path, out_dir: Path, attr_cols: list[str]) -> None:
    """Tokenize one h5ad into out_dir/tokenized.dataset using the gc104M
    token dictionary. Maps gene symbols -> ENSG IDs first (PBMC h5ads store
    gene symbols in var_names; the Geneformer tokenizer keys on ENSG)."""
    import anndata as ad
    from geneformer import TranscriptomeTokenizer

    target = out_dir / "tokenized.dataset"
    if target.exists() and (target / "dataset_info.json").exists():
        print(f"  [tok] skip (exists): {target}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(GENE_NAME_ID_PATH, "rb") as f:
        symbol_to_ensg: dict = pickle.load(f)

    print(f"  [tok] reading {src_h5ad}")
    adata = ad.read_h5ad(src_h5ad)  # full load — largest is ~250MB

    # Map gene symbols -> ENSG; drop unmappable
    var_names = list(adata.var_names)
    keep = np.array([g in symbol_to_ensg for g in var_names])
    n_keep = int(keep.sum())
    print(f"  [tok] mappable genes: {n_keep}/{len(keep)}")
    if n_keep == 0:
        raise RuntimeError("No genes map to ENSG — check gene_name_id_dict_gc104M.pkl")
    adata = adata[:, keep].copy()
    adata.var["ensembl_id"] = [symbol_to_ensg[g] for g in adata.var_names]
    # Geneformer also expects an n_counts obs column for normalization.
    if "n_counts" not in adata.obs.columns:
        X = adata.X
        n_counts = np.asarray(X.sum(axis=1)).ravel()
        adata.obs["n_counts"] = n_counts.astype(np.float64)

    # Stage h5ad in a temp dir; TranscriptomeTokenizer scans a directory.
    with tempfile.TemporaryDirectory(prefix="geneformer_tok_") as td:
        tdp = Path(td)
        staged = tdp / "data.h5ad"
        adata.write_h5ad(staged)
        del adata

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={c: c for c in attr_cols},
            nproc=1,
            chunk_size=512,
            model_input_size=MODEL_INPUT_SIZE,
            special_token=True,
            collapse_gene_ids=True,
            gene_median_file=GENE_MEDIAN_PATH,
            token_dictionary_file=TOKEN_DICT_PATH,
            gene_mapping_file=ENSEMBL_MAP_PATH,
        )
        tk.tokenize_data(
            data_directory=tdp,
            output_directory=str(out_dir),
            output_prefix="tmp",
            file_format="h5ad",
        )
        # tokenize_data writes <prefix>.dataset/; rename to tokenized.dataset
        produced = out_dir / "tmp.dataset"
        if not produced.exists():
            raise RuntimeError(f"Expected {produced} not produced by TranscriptomeTokenizer")
        if target.exists():
            shutil.rmtree(target)
        produced.rename(target)
    print(f"  [tok] -> {target}")


def _attr_cols_for_pbmc(src_h5ad: Path) -> list[str]:
    """Columns to carry into the tokenized dataset (signal labels)."""
    import anndata as ad
    a = ad.read_h5ad(src_h5ad, backed="r")
    cols = ["celltype.l3"] + sorted(
        [c for c in a.obs.columns if c.startswith("prot_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return cols


# -- Fine-tune & embed ----------------------------------------------------

def _fine_tune(train_ds_path: Path, model_out: Path, log_dir: Path, epochs: int) -> None:
    """Continue MLM training the pretrained Geneformer on train_ds_path."""
    import torch
    from datasets import load_from_disk
    from transformers import (
        BertForMaskedLM, TrainingArguments, DataCollatorForLanguageModeling,
        PreTrainedTokenizerFast,
    )
    from geneformer import GeneformerPretrainer

    with open(TOKEN_DICT_PATH, "rb") as f:
        token_dictionary: dict = pickle.load(f)

    train_ds = load_from_disk(str(train_ds_path))
    n_samples = len(train_ds)
    eff_bs = PER_DEVICE_TRAIN_BS * GRAD_ACCUM
    steps_for_epochs = max(1, (n_samples * epochs + eff_bs - 1) // eff_bs)
    capped_steps = min(steps_for_epochs, MAX_STEPS)
    print(f"  [ft] train samples: {n_samples}  steps@{epochs}ep={steps_for_epochs}  -> max_steps={capped_steps}")

    # GeneformerPretrainer requires a pickled length vector for LengthGroupedSampler
    lengths_pkl = train_ds_path.parent / "lengths.pkl"
    if not lengths_pkl.exists():
        lengths = [int(l) for l in train_ds["length"]]
        with open(lengths_pkl, "wb") as f:
            pickle.dump(lengths, f)

    model = BertForMaskedLM.from_pretrained(GENEFORMER_DIR)
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.train()

    model_out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(model_out),
        logging_dir=str(log_dir),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=capped_steps,
        learning_rate=LR,
        weight_decay=1e-3,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_available(),
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=20,
        group_by_length=True,
        length_column_name="length",
        report_to="none",
        dataloader_num_workers=2,
        disable_tqdm=False,
    )

    trainer = GeneformerPretrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        token_dictionary=token_dictionary,
        example_lengths_file=str(lengths_pkl),
    )
    trainer.train()
    trainer.save_model(str(model_out))
    # Persist the tokenizer config too (EmbExtractor reads model_directory)
    with open(model_out / "config.json", "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    del model, trainer
    torch.cuda.empty_cache()


def _embed_test(model_dir: Path, test_ds_path: Path, out_dir: Path,
                emb_label: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run EmbExtractor on the test tokenized dataset.

    Returns (X, Y) where X is (n_cells, hidden_size) and Y is (n_cells, |emb_label|).
    Also writes embeddings.csv and Y_protein_counts.csv to out_dir.
    """
    import torch
    from geneformer import EmbExtractor

    out_dir.mkdir(parents=True, exist_ok=True)
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        gene_emb_style="mean_pool",
        emb_layer=-1,
        forward_batch_size=PER_DEVICE_EVAL_BS,
        nproc=8,
        token_dictionary_file=str(TOKEN_DICT_PATH),
        max_ncells=EMB_MAX_NCELLS,
        emb_label=emb_label,
    )

    embs = embex.extract_embs(
        model_directory=str(model_dir),
        input_data_file=str(test_ds_path),
        output_directory=str(out_dir),
        output_prefix="embeddings",
        output_torch_embs=False,
    )
    # extract_embs returns either a DataFrame or (df, tensor) — normalize
    df = embs[0] if isinstance(embs, tuple) else embs
    # First HIDDEN dims are the cell embeddings; the rest are emb_label cols
    # Hidden size lives in model config — read it back.
    with open(model_dir / "config.json") as f:
        hidden = json.load(f)["hidden_size"]
    arr = df.values
    X = arr[:, :hidden]
    Y = arr[:, hidden:]

    pd.DataFrame(X).to_csv(out_dir / "embeddings.csv", index=False)
    pd.DataFrame(Y, columns=emb_label).to_csv(out_dir / "Y_protein_counts.csv", index=False)
    torch.cuda.empty_cache()
    return X.astype(np.float64), Y.astype(np.float64)


def _compute_mi(X: np.ndarray, Y: np.ndarray, mi_dir: Path, seed: int,
                max_epochs: int = 300) -> float:
    """Estimate MI between cell embeddings X and the multi-dim protein label Y."""
    import torch
    from latentmi import lmi
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    mi_dir.mkdir(parents=True, exist_ok=True)
    pmi, lmi_emb, model = lmi.estimate(
        X, Y,
        validation_split=0.3,
        batch_size=512,
        epochs=max_epochs,
        quiet=False,
    )
    mi = float(np.nanmean(pmi))
    with open(mi_dir / "lmi_mutual_information.txt", "w") as f:
        f.write(f"{mi:.5f}")
    np.save(mi_dir / "lmi_embeddings.npy", lmi_emb)
    torch.save(model.state_dict(), mi_dir / "lmi_model.pt")
    return mi


# -- Per-(size, quality) job (executed in worker subprocess) -------------

def run_one(size: int, quality: float, seed: int, device: int) -> dict:
    """Fine-tune + embed + MI for one (size, quality, seed) cell. One process per GPU."""
    job_name = f"sz{size}_q{quality}_s{seed}"
    t0 = time.time()
    result = {
        "size": int(size),
        "quality": float(quality),
        "seed": int(seed),
        "device": int(device),
        "status": "pending",
        "train_time_s": float("nan"),
        "mi_protein_counts": float("nan"),
        "error": "",
    }

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        _set_seed(seed)

        out_root = OUTPUT_DIR / str(size) / str(quality)
        out_root.mkdir(parents=True, exist_ok=True)

        train_h5ad = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
        test_h5ad = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        assert train_h5ad.exists(), f"missing {train_h5ad}"
        assert test_h5ad.exists(), f"missing {test_h5ad}"

        attr_cols = _attr_cols_for_pbmc(train_h5ad)
        prot_cols = [c for c in attr_cols if c.startswith("prot_")]
        print(f"[{job_name}] attr_cols: celltype.l3 + {len(prot_cols)} prot_*")

        # 1) Tokenize (idempotent). Train per (size, quality); test per quality.
        t = time.time()
        train_tok = out_root  # tokenized.dataset goes inside
        test_tok_root = OUTPUT_DIR / "test" / str(quality)
        _retokenize_h5ad(train_h5ad, train_tok, attr_cols)
        _retokenize_h5ad(test_h5ad, test_tok_root, attr_cols)
        print(f"[{job_name}] tokenize done t={time.time()-t:.0f}s")

        # 2) Fine-tune
        t = time.time()
        model_out = out_root / "model"
        log_dir = out_root / "trainer_logs"
        _fine_tune(train_tok / "tokenized.dataset", model_out, log_dir, EPOCHS)
        print(f"[{job_name}] fine-tune done t={time.time()-t:.0f}s")

        # 3) Embed
        t = time.time()
        X, Y = _embed_test(
            model_out,
            test_tok_root / "tokenized.dataset",
            out_root,
            emb_label=prot_cols,
        )
        print(f"[{job_name}] embed done X={X.shape} Y={Y.shape} t={time.time()-t:.0f}s")

        # 4) MI
        t = time.time()
        mi_dir = out_root / "MI" / str(seed) / "Y_protein_counts"
        mi = _compute_mi(X, Y, mi_dir, seed)
        result["mi_protein_counts"] = mi
        print(f"[{job_name}] MI done mi={mi:.4f} t={time.time()-t:.0f}s")

        # Per-cell config for the sweep dir (mirrors hp_tunning layout)
        (out_root / "config.json").write_text(json.dumps({
            "dataset": DATASET, "size": int(size), "quality": float(quality),
            "seed": int(seed), "epochs": EPOCHS, "lr": LR,
            "per_device_train_bs": PER_DEVICE_TRAIN_BS,
            "grad_accum": GRAD_ACCUM, "warmup_steps": WARMUP_STEPS,
            "model_input_size": MODEL_INPUT_SIZE,
            "pretrained_from": str(GENEFORMER_DIR),
        }, indent=2))

        result["status"] = "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["train_time_s"] = time.time() - t0
        out_root = OUTPUT_DIR / str(size) / str(quality)
        if out_root.exists():
            (out_root / "result.json").write_text(json.dumps(
                {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in result.items()},
                indent=2,
            ))
    return result


# -- Orchestrator --------------------------------------------------------

def detect_gpus() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all required h5ads exist up front
    missing = []
    for size in SIZES:
        for quality in QUALITIES:
            tr = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
            if not tr.exists():
                missing.append(str(tr))
    for quality in QUALITIES:
        te = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        if not te.exists():
            missing.append(str(te))
    if missing:
        print("MISSING h5ads:")
        for m in missing[:20]:
            print(" ", m)
        raise SystemExit(1)
    print(f"All {len(SIZES)} sizes x {len(QUALITIES)} qualities x train+test h5ads verified.")

    gpus = detect_gpus()
    slots = gpus * JOBS_PER_GPU
    max_workers = len(slots)
    print(f"GPUs={gpus} jobs_per_gpu={JOBS_PER_GPU} slots={max_workers}")

    # Pre-tokenize ALL test sets (one per quality) sequentially in the parent
    # process. This is small (10 calls) and avoids a race where multiple
    # workers with the same quality try to tokenize the shared test set.
    print("\nPre-tokenizing test sets (sequential)...")
    for quality in QUALITIES:
        test_h5ad = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        # Build attr_cols from the smallest train h5ad (they all share the schema)
        sample_train = DATA_DIR / DATASET / str(SIZES[0]) / str(quality) / "preprocessed" / "preprocessed.h5ad"
        attr_cols = _attr_cols_for_pbmc(sample_train)
        test_tok_root = OUTPUT_DIR / "test" / str(quality)
        _retokenize_h5ad(test_h5ad, test_tok_root, attr_cols)
    print("Test tokenization complete.\n")

    jobs = []
    for size in SIZES:
        for quality in QUALITIES:
            for seed in SEEDS:
                jobs.append((size, quality, seed))
    # Deterministic shuffle so big and small jobs interleave across GPUs
    random.Random(0).shuffle(jobs)
    total = len(jobs)
    print(f"Total jobs: {total}  (epochs={EPOCHS}, bs={PER_DEVICE_TRAIN_BS}*ga{GRAD_ACCUM})")

    free_slots = list(slots)
    pending = list(jobs)
    results: list[dict] = []
    in_flight: dict = {}

    pbar = tqdm(total=total, desc="geneformer-finetune", unit="cell",
                dynamic_ncols=True, smoothing=0.1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Seed workers
        while pending and free_slots:
            size, quality, seed = pending.pop(0)
            device = free_slots.pop(0)
            fut = ex.submit(run_one, size, quality, seed, device)
            in_flight[fut] = (size, quality, seed, device)
            pbar.write(f"  submit sz{size}_q{quality}_s{seed} on GPU{device}")

        n_ok = n_err = 0
        while in_flight:
            done = next(as_completed(in_flight))
            size, quality, seed, device = in_flight.pop(done)
            try:
                res = done.result()
            except Exception as e:
                res = {"size": int(size), "quality": float(quality),
                       "seed": int(seed), "device": int(device),
                       "status": "error", "train_time_s": float("nan"),
                       "mi_protein_counts": float("nan"),
                       "error": f"{type(e).__name__}: {e}"}
            results.append(res)
            if res["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
            mi = res.get("mi_protein_counts", float("nan"))
            mi_str = f"{mi:.3f}" if not np.isnan(mi) else "nan"
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            pbar.write(f"  done  sz{size}_q{quality}_s{seed} mi={mi_str} "
                       f"t={res['train_time_s']:.0f}s status={res['status']}")
            free_slots.append(device)

            if pending and free_slots:
                size, quality, seed = pending.pop(0)
                device = free_slots.pop(0)
                fut = ex.submit(run_one, size, quality, seed, device)
                in_flight[fut] = (size, quality, seed, device)
                pbar.write(f"  submit sz{size}_q{quality}_s{seed} on GPU{device}")

    pbar.close()

    df = pd.DataFrame(results).sort_values(["size", "quality", "seed"]).reset_index(drop=True)
    out_csv = OUTPUT_DIR / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
