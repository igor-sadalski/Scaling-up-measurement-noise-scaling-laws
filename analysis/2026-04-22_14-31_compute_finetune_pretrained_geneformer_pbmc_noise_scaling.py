"""Fine-tune the pretrained Geneformer-V2-104M on the PBMC noise-scaling grid.

Direct counterpart of
``2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py`` for
Geneformer. Continues MLM pretraining of ``ctheodoris/Geneformer`` (Geneformer-V2
-104M: 1152H / 18L / 18-head / gc104M token dict) on every cell of the PBMC
10 sizes x 10 qualities sweep. For each (size, quality) we:

  1. Tokenize the train h5ad + the per-quality test h5ad with the official
     gc104M dictionary (idempotent -- skip if already done).
  2. MLM continued-pretrain the pretrained checkpoint for ``NUM_EPOCHS`` epochs
     (capped at ``MAX_STEPS`` optimizer steps).
  3. Extract cell embeddings on the test set with ``EmbExtractor``.
  4. Estimate MI between embeddings and the configured signals
     (``celltype.l3`` + ``protein_counts``) via ``latentmi``.

Layout mirrors the STATE fine-tune script so the plotting notebook can reuse
the same collection logic:

    $NOISE_SCALING_OUTPUT_BASE/model_sizing/geneformer/finetune_00/<size>/<quality>/
        config.json
        result.json
        tokenized/          # train tokenized dataset
        model/              # fine-tuned HF model checkpoint
        embeddings.csv
        MI/<seed>/Y_<signal>/lmi_mutual_information.txt

-------------------------------------------------------------------------------
HOW TO DOWNLOAD GENEFORMER-V2-104M AND RUN THIS SCRIPT
-------------------------------------------------------------------------------

1. Activate the main conda env:

     source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling

2. Download the pretrained checkpoint. On first run the script does this
   automatically via ``huggingface_hub.snapshot_download`` into
   ``<repo>/Geneformer/`` (the path STATE expects ``BertForMaskedLM.
   from_pretrained`` to load from). Equivalent manual command:

     huggingface-cli download ctheodoris/Geneformer \\
         --include "config.json" "generation_config.json" \\
                   "model.safetensors" "training_args.bin" \\
                   "geneformer/*.pkl" "geneformer/gene_dictionaries_30m/*.pkl" \\
         --local-dir <repo>/Geneformer

3. Run the sweep (resumes where it left off; skips (size, quality) cells
   whose ``result.json`` status is already "ok"):

     python analysis/2026-04-22_14-31_compute_finetune_pretrained_geneformer_pbmc_noise_scaling.py

4. Plotting: open the companion notebook
   ``2026-04-22_14-31_plotting_finetune_pretrained_geneformer_pbmc_noise_scaling.ipynb``.

Assumptions:
  * ``Experiments.prepare_data(...)`` has already been run for PBMC so the
    preprocessed h5ads exist under
    ``$NOISE_SCALING_DATA_DIR/PBMC/<size>/<quality>/preprocessed/preprocessed.h5ad``
    and ``.../PBMC/test/<quality>/preprocessed/preprocessed.h5ad``.
  * The HuggingFace snapshot contains ``model.safetensors`` + ``config.json`` +
    ``geneformer/*_gc104M.pkl``. If the layout differs the script fails fast.
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
import yaml
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
from scaling_laws.paths import DATA_DIR, GENEFORMER_DIR, OUTPUT_BASE

DATASET = "PBMC"
# Fine-tune only on the largest dataset by default (matches the STATE
# counterpart). Bump this list to sweep more sizes; all downstream code iterates.
SIZES = [100000]
QUALITIES = [
    0.0012346,
    0.0025982,
    0.0054682,
    0.0115083,
    0.02422,
    0.050973,
    0.1072766,
    0.225772,
    0.4751547,
    1.0,
]
SEED = 42

# Pretrained model
HF_MODEL_ID = "ctheodoris/Geneformer"
# We reuse the in-repo ``Geneformer/`` dir as the local snapshot target so
# ``BertForMaskedLM.from_pretrained(GENEFORMER_DIR)`` / dictionary lookups Just Work.
PRETRAINED_DIR = GENEFORMER_DIR
OUTPUT_DIR = OUTPUT_BASE / "model_sizing" / "geneformer"
TRIAL_ID = 0
TRIAL_PREFIX = "finetune"  # -> finetune_00/<size>/<quality>/

# Geneformer gc104M gene dictionaries (shipped inside the HF snapshot at
# <repo>/Geneformer/geneformer/). These are the authoritative vocab for the
# 104M checkpoint -- do NOT mix with the gc95M / gc30M variants.
GENE_DICT_DIR = GENEFORMER_DIR / "geneformer"
TOKEN_DICT_PATH = GENE_DICT_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN_PATH = GENE_DICT_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_NAME_ID_PATH = GENE_DICT_DIR / "gene_name_id_dict_gc104M.pkl"
ENSEMBL_MAP_PATH = GENE_DICT_DIR / "ensembl_mapping_dict_gc104M.pkl"

# Fine-tune hyperparameters (held fixed across the grid)
NUM_EPOCHS = 3
MAX_STEPS = 1500             # hard cap so 100k-cell jobs still finish in <~2h
PER_DEVICE_TRAIN_BS = 1      # 316M model + 2048-token seqs: ~10 GB / GPU
GRAD_ACCUM = 16              # effective batch = 16
PER_DEVICE_EVAL_BS = 2
LR = 5e-5                    # fine-tune LR (vs 1e-3 for from-scratch)
WARMUP_STEPS = 100
MODEL_INPUT_SIZE = 2048      # tokenizer truncation length
EMB_MAX_NCELLS = 5000        # cap test cells used for embedding + MI

# Signals for which to compute MI. Matches the STATE fine-tune counterpart.
SIGNALS = ("celltype.l3", "protein_counts")

# Parallelism. The 316M Geneformer + activations + BF16 at 2048 tokens is
# ~10 GB / GPU -- 1 job/GPU is safe on 24 GB cards; 2 usually fits on 40 GB+.
JOBS_PER_GPU = 1


# -- Pretrained model setup ----------------------------------------------

def download_pretrained(target_dir: Path) -> None:
    """Download the Geneformer HF snapshot into ``target_dir``.

    Idempotent: if ``model.safetensors`` and the gc104M token dictionary are
    already materialised (not Git LFS pointer stubs), skip.
    """
    safetensors = target_dir / "model.safetensors"
    token_dict = TOKEN_DICT_PATH
    # A real safetensors checkpoint is ~1 GB+; a pkl dict ~0.5 MB. Git LFS
    # pointer files are ~100 bytes. Anything >1 MB counts as materialised.
    def _materialised(p: Path, min_bytes: int) -> bool:
        return p.exists() and p.stat().st_size > min_bytes

    if _materialised(safetensors, 100_000_000) and _materialised(token_dict, 100_000):
        print(f"  [pretrained] Weights + dicts already present at {target_dir}; skipping download.")
        return

    print(f"  [pretrained] Downloading {HF_MODEL_ID} -> {target_dir}")
    print(f"  [pretrained] (equivalent manual cmd: huggingface-cli download {HF_MODEL_ID} "
          f"--local-dir {target_dir})")
    target_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_MODEL_ID,
        repo_type="model",
        local_dir=str(target_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "training_args.bin",
            "geneformer/*.pkl",
            "geneformer/gene_dictionaries_30m/*.pkl",
        ],
    )
    # Sanity-check
    if not _materialised(safetensors, 100_000_000):
        raise FileNotFoundError(
            f"Expected materialised model.safetensors under {target_dir} after download."
        )
    if not _materialised(token_dict, 100_000):
        raise FileNotFoundError(
            f"Expected materialised {token_dict} after download -- got {token_dict.stat().st_size} bytes."
        )
    print(f"  [pretrained] Done.")


def verify_pretrained_layout(pretrained_dir: Path) -> dict:
    """Check that ``pretrained_dir`` holds a usable Geneformer-V2-104M snapshot.

    Returns a dict describing the architecture (read from ``config.json``) so
    the per-job worker can log / serialise it.
    """
    cfg_path = pretrained_dir / "config.json"
    weights = pretrained_dir / "model.safetensors"
    missing = [p for p in (cfg_path, weights, TOKEN_DICT_PATH, GENE_MEDIAN_PATH,
                           GENE_NAME_ID_PATH, ENSEMBL_MAP_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Pretrained snapshot incomplete -- missing:\n  " +
            "\n  ".join(str(p) for p in missing)
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    arch = {
        "hidden_size":       int(cfg["hidden_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "vocab_size":        int(cfg["vocab_size"]),
    }
    print(f"  [pretrained] snapshot : {pretrained_dir}")
    print(f"  [pretrained] arch     : {arch}")
    return arch


# -- Re-tokenize PBMC h5ad with the official gc104M dictionary -----------

def _retokenize_h5ad(src_h5ad: Path, out_dir: Path, attr_cols: list[str]) -> None:
    """Tokenize one h5ad into ``out_dir/tokenized.dataset`` using the gc104M
    token dictionary. Maps gene symbols -> ENSG IDs first (PBMC h5ads carry
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
    adata = ad.read_h5ad(src_h5ad)

    var_names = list(adata.var_names)
    keep = np.array([g in symbol_to_ensg for g in var_names])
    n_keep = int(keep.sum())
    print(f"  [tok] mappable genes: {n_keep}/{len(keep)}")
    if n_keep == 0:
        raise RuntimeError("No genes map to ENSG -- check gene_name_id_dict_gc104M.pkl")
    adata = adata[:, keep].copy()
    adata.var["ensembl_id"] = [symbol_to_ensg[g] for g in adata.var_names]
    if "n_counts" not in adata.obs.columns:
        X = adata.X
        n_counts = np.asarray(X.sum(axis=1)).ravel()
        adata.obs["n_counts"] = n_counts.astype(np.float64)

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
        produced = out_dir / "tmp.dataset"
        if not produced.exists():
            raise RuntimeError(f"Expected {produced} not produced by TranscriptomeTokenizer")
        if target.exists():
            shutil.rmtree(target)
        produced.rename(target)
    print(f"  [tok] -> {target}")


def _attr_cols_for_pbmc(src_h5ad: Path) -> list[str]:
    """Columns to carry into the tokenized dataset: celltype label + every prot_*."""
    import anndata as ad
    a = ad.read_h5ad(src_h5ad, backed="r")
    cols = ["celltype.l3"] + sorted(
        [c for c in a.obs.columns if c.startswith("prot_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return cols


# -- Fine-tune & embed ---------------------------------------------------

def _fine_tune(train_ds_path: Path, model_out: Path, log_dir: Path,
               epochs: int, seed: int) -> None:
    """Continue MLM pretraining Geneformer-V2-104M on ``train_ds_path``.

    Mirrors ``Geneformer/examples/pretraining_new_model/pretrain_geneformer_w_deepspeed.py``:
    declarative TrainingArguments + BertForMaskedLM + GeneformerPretrainer,
    one ``trainer.train()`` / ``trainer.save_model()`` pair.
    """
    import torch
    from datasets import load_from_disk
    from transformers import BertForMaskedLM, TrainingArguments
    from geneformer import GeneformerPretrainer

    with open(TOKEN_DICT_PATH, "rb") as f:
        token_dictionary: dict = pickle.load(f)

    train_ds = load_from_disk(str(train_ds_path))
    n_samples = len(train_ds)
    eff_bs = PER_DEVICE_TRAIN_BS * GRAD_ACCUM
    steps_for_epochs = max(1, (n_samples * epochs + eff_bs - 1) // eff_bs)
    capped_steps = min(steps_for_epochs, MAX_STEPS)
    print(f"  [ft] train samples: {n_samples}  steps@{epochs}ep={steps_for_epochs}  -> max_steps={capped_steps}")

    # GeneformerPretrainer's LengthGroupedSampler wants a pickled list of lengths.
    lengths_pkl = train_ds_path.parent / "lengths.pkl"
    if not lengths_pkl.exists():
        with open(lengths_pkl, "wb") as f:
            pickle.dump([int(l) for l in train_ds["length"]], f)

    model = BertForMaskedLM.from_pretrained(PRETRAINED_DIR)
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.train()
    model_out.mkdir(parents=True, exist_ok=True)

    training_args = {
        "output_dir": str(model_out),
        "logging_dir": str(log_dir),
        "do_train": True,
        "do_eval": False,
        "evaluation_strategy": "no",
        "save_strategy": "no",
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BS,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "max_steps": capped_steps,
        "learning_rate": LR,
        "lr_scheduler_type": "linear",
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": 1e-3,
        "bf16": torch.cuda.is_available(),
        "group_by_length": True,
        "length_column_name": "length",
        "logging_steps": 20,
        "report_to": "none",
        "dataloader_num_workers": 2,
        "disable_tqdm": False,
        "seed": seed,
    }

    trainer = GeneformerPretrainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_ds,
        token_dictionary=token_dictionary,
        example_lengths_file=str(lengths_pkl),
    )
    trainer.train()
    trainer.save_model(str(model_out))

    # Persist the raw logs (loss-per-step) as a CSV so the plotting notebook
    # can treat this like the Lightning ``metrics.csv`` from the STATE run.
    try:
        hist = pd.DataFrame(trainer.state.log_history)
        if not hist.empty:
            hist.to_csv(log_dir / "metrics.csv", index=False)
    except Exception as e:
        print(f"  [ft] could not save metrics.csv: {e}")

    del model, trainer
    torch.cuda.empty_cache()


def _embed_test(model_dir: Path, test_ds_path: Path, out_dir: Path,
                emb_label: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run EmbExtractor on the test tokenized dataset.

    Returns (X, Y_by_signal) where X is (n_cells, hidden_size) and Y_by_signal
    maps each signal name in ``SIGNALS`` to an (n_cells, k) array. Also writes
    embeddings.csv and Y_<signal>.csv to ``out_dir``.
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
    df = embs[0] if isinstance(embs, tuple) else embs

    with open(model_dir / "config.json") as f:
        hidden = json.load(f)["hidden_size"]
    arr = df.values
    X = arr[:, :hidden].astype(np.float64)
    label_cols = arr[:, hidden:]

    pd.DataFrame(X).to_csv(out_dir / "embeddings.csv", index=False)

    # Split the label_cols block into per-signal frames using the original
    # ``emb_label`` ordering passed to EmbExtractor.
    y_by_signal: dict[str, np.ndarray] = {}
    for sig in SIGNALS:
        if sig == "celltype.l3":
            idx = [i for i, c in enumerate(emb_label) if c == "celltype.l3"]
        elif sig == "protein_counts":
            idx = [i for i, c in enumerate(emb_label) if c.startswith("prot_")]
        else:
            idx = [i for i, c in enumerate(emb_label) if c == sig]
        if not idx:
            continue
        sub = label_cols[:, idx]
        # celltype.l3 is a string label -> integer-encode for MI
        if sig == "celltype.l3":
            codes = pd.Categorical(sub.ravel()).codes
            y = codes.reshape(-1, 1).astype(np.float64)
            pd.DataFrame({"celltype.l3": sub.ravel()}).to_csv(out_dir / f"Y_{sig}.csv", index=False)
        else:
            y = sub.astype(np.float64)
            pd.DataFrame(y, columns=[emb_label[i] for i in idx]).to_csv(
                out_dir / f"Y_{sig}.csv", index=False
            )
        y_by_signal[sig] = y

    torch.cuda.empty_cache()
    return X, y_by_signal


def _compute_mi(X: np.ndarray, Y: np.ndarray, mi_dir: Path, seed: int,
                max_epochs: int = 300) -> float:
    """Estimate MI between cell embeddings X and label Y via latentmi."""
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

def run_one(size: int, quality: float, device: int, arch: dict) -> dict:
    """Fine-tune + embed + MI for one (size, quality) cell on one GPU."""
    trial_name = f"{TRIAL_PREFIX}_{TRIAL_ID:02d}_{size}_{quality}"
    trial_dir = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / str(size) / str(quality)
    trial_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "trial_id": TRIAL_ID,
        "trial_name": trial_name,
        "size": int(size),
        "quality": float(quality),
        "seed": int(SEED),
        "device": int(device),
        **{f"arch_{k}": v for k, v in arch.items()},
        "status": "pending",
        "train_time_s": float("nan"),
        "error": "",
    }

    # Per-cell config (mirrors state finetune script)
    cfg_doc = {
        "trial_id": TRIAL_ID,
        "trial_name": trial_name,
        "dataset": DATASET,
        "size": int(size),
        "quality": float(quality),
        "seed": SEED,
        "device": int(device),
        "arch": arch,
        "hparams": {
            "num_epochs": NUM_EPOCHS,
            "max_steps": MAX_STEPS,
            "lr": LR,
            "per_device_train_bs": PER_DEVICE_TRAIN_BS,
            "grad_accum": GRAD_ACCUM,
            "warmup_steps": WARMUP_STEPS,
            "model_input_size": MODEL_INPUT_SIZE,
            "emb_max_ncells": EMB_MAX_NCELLS,
        },
        "pretrained": {
            "hf_model_id": HF_MODEL_ID,
            "pretrained_dir": str(PRETRAINED_DIR),
        },
    }
    (trial_dir / "config.json").write_text(json.dumps(cfg_doc, indent=2, default=str))

    t0 = time.time()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        # Seeds
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        train_h5ad = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
        test_h5ad = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        assert train_h5ad.exists(), f"missing {train_h5ad}"
        assert test_h5ad.exists(), f"missing {test_h5ad}"

        attr_cols = _attr_cols_for_pbmc(train_h5ad)
        prot_cols = [c for c in attr_cols if c.startswith("prot_")]
        print(f"[{trial_name}] attr_cols: celltype.l3 + {len(prot_cols)} prot_*")

        # 1) Tokenize (idempotent). Train per (size, quality); test shared per quality.
        t = time.time()
        train_tok_root = trial_dir / "tokenized"
        test_tok_root = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / "test" / str(quality)
        _retokenize_h5ad(train_h5ad, train_tok_root, attr_cols)
        _retokenize_h5ad(test_h5ad, test_tok_root, attr_cols)
        print(f"[{trial_name}] tokenize DONE t={time.time()-t:.0f}s")

        # 2) Fine-tune
        t = time.time()
        model_out = trial_dir / "model"
        log_dir = trial_dir / "trainer_logs"
        _fine_tune(train_tok_root / "tokenized.dataset", model_out, log_dir, NUM_EPOCHS, SEED)
        print(f"[{trial_name}] fine-tune DONE t={time.time()-t:.0f}s")

        # 3) Embed
        t = time.time()
        X, y_by_signal = _embed_test(
            model_out,
            test_tok_root / "tokenized.dataset",
            trial_dir,
            emb_label=attr_cols,
        )
        print(f"[{trial_name}] embed DONE X={X.shape} t={time.time()-t:.0f}s")

        # 4) MI -- one estimate per signal.
        for sig, Y in y_by_signal.items():
            t = time.time()
            mi_dir = trial_dir / "MI" / str(SEED) / f"Y_{sig}"
            mi = _compute_mi(X, Y, mi_dir, SEED)
            result[f"mi_{sig}"] = mi
            print(f"[{trial_name}] MI[{sig}] DONE mi={mi:.4f} t={time.time()-t:.0f}s")

        result["status"] = "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["train_time_s"] = time.time() - t0
        serialised = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in result.items()}
        (trial_dir / "result.json").write_text(json.dumps(serialised, indent=2, default=str))
        print(f"[{trial_name}] status={result['status']} t={result['train_time_s']:.0f}s", flush=True)
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

    # 1. Pretrained model: download + verify
    download_pretrained(PRETRAINED_DIR)
    arch = verify_pretrained_layout(PRETRAINED_DIR)

    # 2. Verify the PBMC noise-scaling grid is ready on disk.
    missing = []
    for size in SIZES:
        for quality in QUALITIES:
            tr = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
            if not tr.exists():
                missing.append(tr)
    for quality in QUALITIES:
        te = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        if not te.exists():
            missing.append(te)
    if missing:
        print(f"MISSING {len(missing)} h5ads -- run Experiments.prepare_data() first. First 10:")
        for p in missing[:10]:
            print(f"  {p}")
        sys.exit(1)
    print(f"All {len(SIZES)} x {len(QUALITIES)} train h5ads + {len(QUALITIES)} test h5ads verified.")

    # 3. Pre-tokenize shared test sets sequentially (one per quality) in the
    # parent so parallel workers with the same quality don't race.
    print("\nPre-tokenizing test sets (sequential)...")
    for quality in tqdm(QUALITIES, desc="pre-tok test", unit="quality"):
        test_h5ad = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
        sample_train = DATA_DIR / DATASET / str(SIZES[0]) / str(quality) / "preprocessed" / "preprocessed.h5ad"
        attr_cols = _attr_cols_for_pbmc(sample_train)
        test_tok_root = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / "test" / str(quality)
        _retokenize_h5ad(test_h5ad, test_tok_root, attr_cols)
    print("Test tokenization complete.\n")

    # 4. Schedule the fine-tune grid across GPUs.
    gpus = detect_gpus()
    slots = gpus * JOBS_PER_GPU
    max_workers = max(1, len(slots))
    print(f"GPUs={gpus} JOBS_PER_GPU={JOBS_PER_GPU} slots={max_workers}")

    jobs = [(size, quality) for size in SIZES for quality in QUALITIES]
    # Skip cells that already completed (resume-friendly).
    pending = []
    for size, quality in jobs:
        trial_dir = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / str(size) / str(quality)
        res = trial_dir / "result.json"
        if res.exists():
            try:
                if json.loads(res.read_text()).get("status") == "ok":
                    continue
            except Exception:
                pass
        pending.append((size, quality))
    random.Random(SEED).shuffle(pending)
    total = len(pending)
    print(f"Scheduling {total} / {len(jobs)} cells ({len(jobs) - total} already complete).")
    if total == 0:
        print("Nothing to do.")
        _write_summary()
        return

    free_slots = list(slots)
    in_flight: dict = {}
    results: list[dict] = []
    pbar = tqdm(total=total, desc="finetune Geneformer-V2-104M", unit="cell",
                dynamic_ncols=True, smoothing=0.1)
    n_ok = n_err = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while pending and free_slots:
            size, quality = pending.pop(0)
            device = free_slots.pop(0)
            fut = ex.submit(run_one, size, quality, device, arch)
            in_flight[fut] = (size, quality, device)
            pbar.write(f"  submit size={size} q={quality} GPU={device}")

        while in_flight:
            done = next(as_completed(in_flight))
            size, quality, device = in_flight.pop(done)
            try:
                res = done.result()
            except Exception as e:
                res = {"size": int(size), "quality": float(quality), "seed": SEED,
                       "device": int(device), "status": "error",
                       "train_time_s": float("nan"),
                       "error": f"{type(e).__name__}: {e}"}
            results.append(res)
            if res.get("status") == "ok":
                n_ok += 1
            else:
                n_err += 1
            mi_str = " ".join(f"{k}={v:.3f}" for k, v in res.items()
                              if k.startswith("mi_") and isinstance(v, (int, float))
                              and not (isinstance(v, float) and np.isnan(v)))
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            pbar.write(f"  done   size={size} q={quality} status={res.get('status')} {mi_str} "
                       f"t={res.get('train_time_s', 0):.0f}s")
            free_slots.append(device)
            if pending and free_slots:
                size, quality = pending.pop(0)
                device = free_slots.pop(0)
                fut = ex.submit(run_one, size, quality, device, arch)
                in_flight[fut] = (size, quality, device)
                pbar.write(f"  submit size={size} q={quality} GPU={device}")
    pbar.close()

    _write_summary()


def _write_summary():
    """Aggregate per-cell result.json into sweep_results.csv + per-trial YAML."""
    rows = []
    trial_root = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}"
    if not trial_root.is_dir():
        return
    for size_dir in sorted(p for p in trial_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for q_dir in sorted(p for p in size_dir.iterdir() if p.is_dir()):
            res_path = q_dir / "result.json"
            if not res_path.exists():
                continue
            try:
                rows.append(json.loads(res_path.read_text()))
            except Exception:
                pass
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["size", "quality"]).reset_index(drop=True)
    out_csv = OUTPUT_DIR / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSummary -> {out_csv}")
    ok = df[df["status"] == "ok"]
    mi_cols = [c for c in df.columns if c.startswith("mi_")]
    if len(ok) and mi_cols:
        print(ok.groupby("size")[mi_cols].mean().to_string())

    per_trial = {"trial_id": TRIAL_ID, "hf_model_id": HF_MODEL_ID, "results": {}}
    for _, r in df.iterrows():
        per_trial["results"].setdefault(int(r["size"]), {})[float(r["quality"])] = {
            "status": r.get("status"),
            "train_time_s": r.get("train_time_s"),
            "mi": {c.removeprefix("mi_"): r[c] for c in mi_cols if pd.notna(r.get(c))},
        }
    with open(OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}.yaml", "w") as f:
        yaml.dump(per_trial, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
