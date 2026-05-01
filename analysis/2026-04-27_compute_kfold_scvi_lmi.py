"""Train one SCVI per (fold, train_run), embed val pool, compute latentmi MI.

For each fold directory `kfold/{K}-fold/` produced by
`2026-04-27_build_kfold_splits.py` (8-fold CV with 50% val across the 16
author_experimental_ids), the fold's `train/` holds 8 per-run h5ads
(`{run}_edist{value}.h5ad`) and `val/` holds 8 per-run h5ads. This script
trains one SCVI model per training run on its single ~10k-cell h5ad,
embeds the fold's ~80k-cell val pool, and runs `latentmi` against
one-hot `author_day`. Each (fold, run_id) is one worker doing
train -> embed -> LMI sequentially in-process; workers run JOBS_PER_GPU
per GPU (default 2) on the 8 visible GPUs via a spawn ProcessPoolExecutor.
Hyperparameters match `scaling_laws/algo/scvi.py:SCVI.train` and
`scaling_laws/algo/abc.py:BaseAlgorithm.mutual_information`.

Reads:
  data_local/other/batch_effects/kfold/{K}-fold/train/{run}_edist{value}.h5ad
  data_local/other/batch_effects/kfold/{K}-fold/val/{run}.h5ad

Writes:
  data_local/other/batch_effects/kfold/scvi/{K}-fold/{run_id}/model/
  data_local/other/batch_effects/kfold/scvi/{K}-fold/{run_id}/val_embeddings.csv
  data_local/other/batch_effects/kfold/scvi/{K}-fold/{run_id}/val_obs.csv
  data_local/other/batch_effects/kfold/scvi/{K}-fold/{run_id}/lmi/seed_42/{lmi_mutual_information.txt, lmi_embeddings.npy, lmi_model.pt}
  data_local/other/batch_effects/kfold/scvi/{K}-fold/{run_id}/summary.json
  data_local/other/batch_effects/kfold/scvi/run.log
  data_local/other/batch_effects/kfold/scvi/worker_logs/{K}-fold_{run_id}.log
  analysis/final_results/kfold_scvi_lmi.csv
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
KFOLD_DIR = Path(
    "/home/igor/igor_repos/scaling_laws/data_local/other/batch_effects/kfold"
)
SCVI_OUT_DIR = KFOLD_DIR / "scvi"
WORKER_LOG_DIR = SCVI_OUT_DIR / "worker_logs"
RUN_LOG = SCVI_OUT_DIR / "run.log"

FINAL_RESULTS_DIR = SCRIPT_PATH.parent / "final_results"
SUMMARY_CSV = FINAL_RESULTS_DIR / "kfold_scvi_lmi.csv"

SIGNAL_COLUMN = "author_day"
LMI_SEED = 42
SCVI_MAX_EPOCHS = 25
SCVI_BATCH_SIZE = 512
LMI_MAX_EPOCHS = 300
# Physical GPU indices to use. Set to a subset (e.g. [1, 4]) when other users
# occupy some GPUs. Workers round-robin across this list.
GPU_INDICES = [1, 2, 4]
N_GPUS = len(GPU_INDICES)
JOBS_PER_GPU = 3

# Limit which fold subdirs to process (None = all folds discovered on disk).
FOLD_WHITELIST: set[str] | None = None

EDIST_RE = re.compile(r"^(?P<run>.+)_edist(?P<v>\d+\.\d+)\.h5ad$")
FOLD_RE = re.compile(r"^(?P<k>\d+)-fold$")


class Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _setup_main_logging() -> None:
    SCVI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(RUN_LOG, "w")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"Logging to {RUN_LOG}")


def discover_jobs() -> list[dict]:
    """Walk KFOLD_DIR and emit one job dict per (fold, train_run)."""
    jobs: list[dict] = []
    fold_dirs = sorted(
        (p for p in KFOLD_DIR.glob("*-fold") if p.is_dir() and FOLD_RE.match(p.name)),
        key=lambda p: int(FOLD_RE.match(p.name).group("k")),
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"no '*-fold' directories under {KFOLD_DIR}. Run "
            f"2026-04-27_build_kfold_splits.py first."
        )
    for fold_dir in fold_dirs:
        fold_label = fold_dir.name
        if FOLD_WHITELIST is not None and fold_label not in FOLD_WHITELIST:
            print(f"[discover] {fold_label}: not in whitelist {sorted(FOLD_WHITELIST)} -- skipping")
            continue
        val_dir = fold_dir / "val"
        train_dir = fold_dir / "train"
        val_paths = sorted(val_dir.glob("*.h5ad")) if val_dir.is_dir() else []
        if not val_paths:
            print(f"[discover] {fold_label}: no val h5ads under {val_dir} -- skipping")
            continue
        train_files = sorted(train_dir.glob("*_edist*.h5ad")) if train_dir.is_dir() else []
        if not train_files:
            print(f"[discover] {fold_label}: no train h5ads under {train_dir} -- skipping")
            continue
        for t in train_files:
            m = EDIST_RE.match(t.name)
            if not m:
                print(f"[discover] {fold_label}: skipping non-matching {t.name}")
                continue
            jobs.append(
                {
                    "fold_label": fold_label,
                    "run_id": m.group("run"),
                    "train_h5ad": str(t),
                    "val_paths": [str(p) for p in val_paths],
                    "edist": float(m.group("v")),
                }
            )
    return jobs


def job_out_dir(fold_label: str, run_id: str) -> Path:
    return SCVI_OUT_DIR / fold_label / run_id


def already_done(fold_label: str, run_id: str) -> bool:
    summary = job_out_dir(fold_label, run_id) / "summary.json"
    if not summary.exists():
        return False
    try:
        d = json.loads(summary.read_text())
        return d.get("lmi") is not None
    except Exception:
        return False


def process_job(
    fold_label: str,
    run_id: str,
    train_h5ad: str,
    val_paths: list[str],
    edist: float,
    gpu_id: int,
) -> dict:
    """Worker: pin GPU before importing torch/scvi, train, embed, run LMI."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    worker_log = WORKER_LOG_DIR / f"{fold_label}_{run_id}.log"
    wf = open(worker_log, "w", buffering=1)
    # Tee to the parent terminal so verbose progress bars stream live, while
    # also keeping a clean per-worker log file.
    sys.stdout = Tee(sys.__stdout__, wf)
    sys.stderr = Tee(sys.__stderr__, wf)
    tag = f"[{fold_label}/{run_id}]"
    print(
        f"{tag} gpu={gpu_id} train={Path(train_h5ad).name} "
        f"val_files={[Path(v).name for v in val_paths]} edist={edist:.6f}"
    )

    import numpy as np
    import torch
    import anndata as ad
    import scvi
    from latentmi import lmi

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU is required but CUDA is not available "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )
    torch.cuda.set_device(0)

    out_dir = job_out_dir(fold_label, run_id)
    model_dir = out_dir / "model"
    lmi_dir = out_dir / "lmi" / f"seed_{LMI_SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    lmi_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train SCVI on the single-run train h5ad ----
    t0 = time.time()
    print(f"{tag} reading train h5ad")
    adata_train = ad.read_h5ad(train_h5ad)
    n_train_cells = int(adata_train.n_obs)
    print(f"{tag} train shape={adata_train.shape}")

    scvi.model.SCVI.setup_anndata(adata_train)
    vae = scvi.model.SCVI(
        adata_train,
        n_hidden=512,
        n_latent=16,
        n_layers=1,
        dropout_rate=0.1,
        dispersion="gene",
        gene_likelihood="zinb",
        latent_distribution="normal",
    )
    train_plan = {
        "lr": 1e-3,
        "weight_decay": 1e-06,
        "eps": 0.01,
        "n_epochs_kl_warmup": 1,
        "reduce_lr_on_plateau": True,
        "lr_scheduler_metric": "elbo_validation",
        "lr_min": 1e-6,
        "max_kl_weight": 1.0,
        "min_kl_weight": 0.0,
        "compile": False,
    }
    print(f"{tag} training SCVI (max_epochs={SCVI_MAX_EPOCHS})")
    # log_every_n_steps=1 + check_val_every_n_epoch=1 + early_stopping=False
    # makes every metric in vae.history per-epoch indexed (epoch 0..max_epochs-1)
    # with no NaN gaps or step/epoch index aliasing.
    vae.train(
        accelerator="gpu",
        devices=1,
        train_size=0.8,
        validation_size=0.2,
        shuffle_set_split=True,
        load_sparse_tensor=False,
        batch_size=SCVI_BATCH_SIZE,
        plan_kwargs=train_plan,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
        early_stopping=False,
        max_epochs=SCVI_MAX_EPOCHS,
    )
    vae.save(str(model_dir), overwrite=True, save_anndata=False)
    train_elapsed = time.time() - t0

    # Persist only the per-epoch train/val ELBO. vae.history mixes per-step
    # metrics (lr-Adam, kl_weight) with per-epoch ones, both labeled "epoch",
    # so concatenating everything misaligns the index. Just save what we plot.
    history_dict = vae.history
    train_df = history_dict.get("elbo_train")
    val_df = history_dict.get("elbo_validation")
    if train_df is not None and val_df is not None:
        history_combined = pd.DataFrame({
            "train_loss": train_df.iloc[:, 0],
            "val_loss": val_df.iloc[:, 0],
        })
        history_combined.index.name = "epoch"
        history_combined.to_csv(out_dir / "training_history.csv", index=True)
        print(
            f"{tag} wrote training history ({len(history_combined)} epochs) "
            f"-> {out_dir / 'training_history.csv'}"
        )
    else:
        print(f"{tag} WARNING: elbo_train/elbo_validation missing from vae.history")
    print(f"{tag} SCVI trained + saved in {train_elapsed:.1f}s")

    # ---- 2. Build val AnnData by concatenating per-id h5ads ----
    t0 = time.time()
    val_per_id = [ad.read_h5ad(p) for p in val_paths]
    adata_val = ad.concat(val_per_id, join="inner")
    if SIGNAL_COLUMN not in adata_val.obs.columns:
        raise KeyError(
            f"{SIGNAL_COLUMN!r} not in val obs columns: "
            f"{list(adata_val.obs.columns)[:8]}..."
        )
    common_vars = adata_train.var_names.intersection(adata_val.var_names)
    if len(common_vars) != adata_train.n_vars:
        print(
            f"{tag} WARNING: train n_vars={adata_train.n_vars}, "
            f"val n_vars={adata_val.n_vars}, common={len(common_vars)}"
        )
    adata_val = adata_val[:, adata_train.var_names].copy()
    n_val_cells = int(adata_val.n_obs)
    print(f"{tag} val shape={adata_val.shape} (built in {time.time()-t0:.1f}s)")

    del adata_train, val_per_id

    # ---- 3. Embed val with the trained SCVI model ----
    t0 = time.time()
    vae_loaded = scvi.model.SCVI.load(dir_path=str(model_dir), adata=adata_val)
    latent = vae_loaded.get_latent_representation(adata_val)
    pd.DataFrame(latent).to_csv(out_dir / "val_embeddings.csv", index=False)
    val_obs = adata_val.obs[[SIGNAL_COLUMN]].astype(str).reset_index(drop=True)
    val_obs.to_csv(out_dir / "val_obs.csv", index=False)
    print(
        f"{tag} embedded val (latent={latent.shape}) in {time.time()-t0:.1f}s"
    )

    # ---- 4. LMI(X=latent, Y=one-hot(author_day)) ----
    torch.cuda.set_device(0)
    embeddings_df = pd.read_csv(out_dir / "val_embeddings.csv")
    X = embeddings_df.values.astype("float64")
    signal_data = pd.read_csv(out_dir / "val_obs.csv").values
    Y = pd.get_dummies(signal_data.ravel()).values.astype("float32")
    print(f"{tag} running latentmi (X={X.shape}, Y={Y.shape}, seed={LMI_SEED})")
    t0 = time.time()
    pmi, lmi_embeddings, model = lmi.estimate(
        X,
        Y,
        validation_split=0.3,
        batch_size=512,
        epochs=LMI_MAX_EPOCHS,
        quiet=False,
    )
    mi_value = float(np.nanmean(pmi))
    print(
        f"{tag} LMI={mi_value:.5f} ({time.time()-t0:.1f}s) edist={edist:.5f}"
    )

    (lmi_dir / "lmi_mutual_information.txt").write_text(f"{mi_value:.5f}")
    np.save(lmi_dir / "lmi_embeddings.npy", lmi_embeddings)
    torch.save(model.state_dict(), lmi_dir / "lmi_model.pt")

    # ---- 5. Summary ----
    val_ids = [Path(p).stem for p in val_paths]
    summary = {
        "fold": fold_label,
        "run_id": run_id,
        "gpu_id": gpu_id,
        "train_h5ad": str(train_h5ad),
        "val_files": [str(p) for p in val_paths],
        "val_ids": val_ids,
        "edist_train_val": edist,
        "n_train_cells": n_train_cells,
        "n_val_cells": n_val_cells,
        "lmi": mi_value,
        "lmi_seed": LMI_SEED,
        "scvi_max_epochs": SCVI_MAX_EPOCHS,
        "scvi_train_seconds": train_elapsed,
        "lmi_max_epochs": LMI_MAX_EPOCHS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{tag} wrote summary -> {out_dir / 'summary.json'}")
    return summary


def collect_summaries(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for j in jobs:
        s_path = job_out_dir(j["fold_label"], j["run_id"]) / "summary.json"
        if not s_path.exists():
            continue
        d = json.loads(s_path.read_text())
        rows.append(
            {
                "fold": d["fold"],
                "run_id": d["run_id"],
                "val_ids": ",".join(d.get("val_ids", [])),
                "edist_train_val": d["edist_train_val"],
                "lmi": d["lmi"],
                "lmi_seed": d["lmi_seed"],
                "n_train_cells": d.get("n_train_cells"),
                "n_val_cells": d.get("n_val_cells"),
                "scvi_train_seconds": d.get("scvi_train_seconds"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["fold", "edist_train_val"]).reset_index(drop=True)


def main() -> None:
    _setup_main_logging()
    SCVI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs()
    by_fold: dict[str, int] = {}
    for j in jobs:
        by_fold[j["fold_label"]] = by_fold.get(j["fold_label"], 0) + 1
    print(f"Discovered {len(jobs)} (fold, run) jobs across {len(by_fold)} folds:")
    for f, n in sorted(by_fold.items()):
        print(f"  {f}: {n} train runs")

    pending = [j for j in jobs if not already_done(j["fold_label"], j["run_id"])]
    print(f"{len(pending)}/{len(jobs)} jobs pending")

    if pending:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(pending), N_GPUS * JOBS_PER_GPU), mp_context=ctx
        ) as ex:
            futures = {}
            for i, j in enumerate(pending):
                gpu_id = GPU_INDICES[i % N_GPUS]
                fut = ex.submit(
                    process_job,
                    j["fold_label"],
                    j["run_id"],
                    j["train_h5ad"],
                    list(j["val_paths"]),
                    float(j["edist"]),
                    gpu_id,
                )
                futures[fut] = (j["fold_label"], j["run_id"], gpu_id)

            with tqdm(total=len(futures), desc="jobs") as pbar:
                for fut in as_completed(futures):
                    fold_label, run_id, gpu_id = futures[fut]
                    try:
                        s = fut.result()
                        tqdm.write(
                            f"[main] {fold_label}/{run_id} gpu={gpu_id} "
                            f"lmi={s['lmi']:.5f} edist={s['edist_train_val']:.5f}"
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        tqdm.write(
                            f"[main] FATAL {fold_label}/{run_id} gpu={gpu_id}: {e}\n{tb}"
                        )
                    pbar.update(1)

    df = collect_summaries(jobs)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote summary CSV ({len(df)} rows) to {SUMMARY_CSV}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
