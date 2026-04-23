"""Fine-tune the pretrained SE-100M STATE model on the PBMC noise-scaling grid.

This script runs a short (1-epoch) fine-tune of the Arc Institute SE-100M
embedding model across the full PBMC 10-sizes x 10-qualities grid. For each
(size, quality) we:

  1. Re-run ``state emb preprocess`` against SE-100M's gene-embedding vocabulary
     so our ds_emb_mapping / valid_genes_masks line up with the pretrained
     pe_embedding (our existing ``data/PBMC/.../preprocessed/state_data/`` uses
     ``merged_esm_embeddings.pt``; SE-100M ships its own gene-ordering).
  2. Copy a pre-stripped ``finetune_init.ckpt`` (pretrained weights only; no
     trainer / optimizer / scheduler state) to the spot STATE's trainer expects
     for auto-resume (``<checkpoint_dir>/state_<profile>/last.ckpt``). This
     triggers Lightning to load weights but NOT bypass the fresh optimizer,
     scheduler, or epoch counter built from the current Hydra config.
  3. Run ``state emb fit`` with ``experiment.num_epochs=1``,
     ``optimizer.max_lr=1e-5``, ``optimizer.reset_lr_on_restart=true``. Early
     stopping (patience=5, monitor=validation/val_loss) stays enabled so tiny
     cells terminate gracefully.
  4. ``state emb transform`` to embed the test set, then LMI MI against
     ``celltype.l3`` and ``protein_counts``.

Outputs go to ``$NOISE_SCALING_OUTPUT_BASE/finetunning_state/`` (user-chosen
name). The layout mirrors ``model_sizing/`` so the plotting notebook can reuse
the same collection logic.

-------------------------------------------------------------------------------
HOW TO DOWNLOAD SE-100M AND RUN THIS SCRIPT  (for future maintainers)
-------------------------------------------------------------------------------

1. Activate the main conda env (the ``modeling`` env has ``huggingface_hub``;
   the ``state`` env is invoked internally as a subprocess):

     source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling

2. Download the pretrained model from HuggingFace. The script does this
   automatically on first run via ``huggingface_hub.snapshot_download``, but
   you can also do it manually:

     huggingface-cli download arcinstitute/SE-100M \\
         --local-dir ~/noise_scaling/data/other/finetunning_state/pretrained/SE-100M

3. Run the sweep (resumes where it left off; skips (size, quality) cells
   whose ``result.json`` status is already "ok"):

     python analysis/2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py

4. To bump parallelism, edit ``JOBS_PER_GPU`` at the top of this file. SE-100M
   at batch_size=64 fits comfortably in ~10-12GB per job — two jobs per 24GB
   GPU usually works, but watch ``nvidia-smi`` on the first few submissions.

5. Plotting: open the companion notebook
   ``2026-04-21_14-00_plotting_finetune_pretrained_state_pbmc_noise_scaling.ipynb``.

Assumptions:
  * ``Experiments.prepare_state_data(...)`` has already been run for PBMC so
    the preprocessed h5ads + CSV manifests exist under
    ``$NOISE_SCALING_DATA_DIR/PBMC/<size>/<quality>/preprocessed/``.
  * The SE-100M HuggingFace snapshot contains one ``*.ckpt`` file, a
    ``config.yaml`` describing the model architecture, and (either in the
    snapshot or reachable via its config) an ``all_embeddings*.pt`` gene
    embedding file. If the layout differs, the script fails fast with a
    message pointing at what it could not find.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
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


# ── Configuration ───────────────────────────────────────────────────────
from scaling_laws.paths import DATA_DIR, OUTPUT_BASE

DATASET = "PBMC"
SIZES = [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000]
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
HF_MODEL_ID = "arcinstitute/SE-100M"
OUTPUT_DIR = OUTPUT_BASE / "finetunning_state"
PRETRAINED_DIR = OUTPUT_DIR / "pretrained" / "SE-100M"
TRIAL_ID = 0
TRIAL_PREFIX = "finetune"  # -> finetune_00/<size>/<quality>/

# Fine-tune hyperparameters. The "base" epoch/LR settings are scaled per (size,
# quality) cell in ``run_one`` below — see the grid-aware block there.
#
# Prior values (1 epoch, 1e-5 max_lr, dropout 0.1) were suppressing MI:
#  * 1e-5 is the SE-100M *pretraining* LR (state-defaults.yaml:236) — not
#    "10x below", so step 0 of fine-tune re-entered pretrain regime and
#    overwrote the representation before the MI probe could benefit.
#  * 1 epoch at val_check_interval=1000 never gave StepBasedEarlyStopping
#    (emb/train/trainer.py:153-161, patience=5) enough val windows to fire
#    on cells < 10k cells; the sweep trained blind there.
NUM_EPOCHS_BASE = 3          # fixed 3 epochs per user request (was 10, was 1)
FINETUNE_MAX_LR = 1e-6       # was 1e-5 — now actually 10× below pretrain
VAL_CHECK_INTERVAL = 1000    # cap; run_one clamps to batches_per_epoch//4
BATCH_SIZE = 64
DROPOUT = 0.15               # was 0.1 — more regularisation at low q/size
# Effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS. SE-100M pretraining used
# 128 × 8 = 1024 (state-defaults.yaml:215, 241). We match 1024 exactly via
# 64 × 16 — 64 mini-batch stays within 80 GB VRAM at pad_length=2048;
# 128 mini-batch would OOM given the ~69 GB we already see per job.
GRAD_ACCUM_STEPS = 16        # was 4 — now matches SE-100M pretrain effective batch
# SE-100M pretraining used dataset_correction=true with a 14418-class classifier
# (see PRETRAINED_DIR/config.yaml:dataset.scbasecamp-cellxgene-tahoe-filtered.num_datasets).
# We keep both settings during fine-tune so the pretrained state_dict loads clean.
#
# Why not flip dataset_correction=false? It changes binary_decoder's first
# Linear input dim from (output_dim + d_model + 11) → (output_dim + d_model + 1)
# (emb/nn/model.py:118-125), a shape mismatch that load_state_dict refuses even
# with strict=False. Handling it properly requires either in-process loading
# with ``StateEmbeddingModel.load_from_checkpoint(..., strict=False)`` (the
# idiom in emb/finetune_decoder.py:94) or explicit state-dict surgery on
# ``binary_decoder.0.intermediate_dense.weight``. Both out of scope for now.
PRETRAIN_NUM_DATASETS = 14418

# Parallelism: ONE constant to bump if you have VRAM headroom.
#
# SE-100M at batch_size=64 + pad_length=2048 uses ~10-12 GB per job. On a 24 GB
# GPU, 2/GPU is usually fine (try it and watch ``nvidia-smi``). On an 80 GB
# A100, 4-6/GPU works.
JOBS_PER_GPU = 1


# ── Pretrained model setup ──────────────────────────────────────────────

def download_pretrained(target_dir: Path) -> None:
    """Download the SE-100M HuggingFace snapshot to ``target_dir``.

    Idempotent: if ``target_dir`` already contains any ``*.ckpt`` file, skip.
    """
    if target_dir.exists() and any(target_dir.rglob("*.ckpt")):
        print(f"  [pretrained] Snapshot already present at {target_dir}; skipping download.")
        return

    print(f"  [pretrained] Downloading {HF_MODEL_ID} -> {target_dir}")
    print(f"  [pretrained] (equivalent manual cmd: "
          f"huggingface-cli download {HF_MODEL_ID} --local-dir {target_dir})")
    target_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_MODEL_ID,
        repo_type="model",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"  [pretrained] Done.")


def locate_pretrained(pretrained_dir: Path) -> dict:
    """Return {'ckpt', 'config', 'embeddings', 'arch'} from the HF snapshot.

    Policy:
      * ckpt: prefer ``*.ckpt``, fall back to ``model.safetensors`` /
        ``*.safetensors`` (the format SE-100M ships in on HuggingFace).
      * config: ``config.yaml`` at the root, or the first ``*.yaml`` found.
      * embeddings: any ``all_embeddings*.pt``, ``protein_embeddings*.pt``,
        ``*gene_symbol_to_embedding*.pt`` or ``*gene_embeddings*.pt`` in the
        snapshot; or the path referenced by the config's
        ``embeddings.<current>.all_embeddings`` if resolvable locally.
      * arch: parsed from the config's ``model`` section.
    """
    # Exclude our own derived ``finetune_init.ckpt`` — it is the OUTPUT of
    # ``build_finetune_init_ckpt``, not a source. Picking it up here would feed
    # the derived file back into itself on subsequent runs.
    ckpts = [p for p in sorted(pretrained_dir.rglob("*.ckpt"),
                               key=lambda p: p.stat().st_size, reverse=True)
             if p.name != "finetune_init.ckpt"]
    if not ckpts:
        ckpts = sorted(pretrained_dir.rglob("*.safetensors"),
                       key=lambda p: p.stat().st_size, reverse=True)
    if not ckpts:
        raise FileNotFoundError(
            f"No *.ckpt or *.safetensors under {pretrained_dir} — HF download failed?"
        )
    ckpt = ckpts[0]

    # Prefer ``config.yaml`` at root; fall back to any yaml in the snapshot.
    cfg_path = pretrained_dir / "config.yaml"
    if not cfg_path.exists():
        yamls = list(pretrained_dir.rglob("*.yaml")) + list(pretrained_dir.rglob("*.yml"))
        if not yamls:
            raise FileNotFoundError(
                f"No config.yaml under {pretrained_dir}. The SE-100M snapshot is "
                f"expected to ship a config.yaml with the ``model`` section."
            )
        cfg_path = yamls[0]

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Gene embeddings file: accept the common Arc Institute naming conventions.
    # SE-100M on HuggingFace ships ``protein_embeddings.pt`` (dict of gene_symbol
    # -> ESM-2 tensor), which is functionally the ``all_embeddings`` input that
    # ``state emb preprocess --all-embeddings`` expects.
    emb_candidates = (
        list(pretrained_dir.rglob("all_embeddings*.pt"))
        + list(pretrained_dir.rglob("protein_embeddings*.pt"))
        + list(pretrained_dir.rglob("*gene_symbol_to_embedding*.pt"))
        + list(pretrained_dir.rglob("*gene_embeddings*.pt"))
    )
    embeddings_path = None
    if emb_candidates:
        embeddings_path = emb_candidates[0]
    else:
        # Fall back to the path referenced by the snapshot's config (in case the
        # snapshot ships a small config but expects embeddings elsewhere).
        try:
            current = cfg["embeddings"]["current"]
            ref = Path(cfg["embeddings"][current]["all_embeddings"])
            if ref.exists():
                embeddings_path = ref
        except Exception:
            pass

    if embeddings_path is None:
        raise FileNotFoundError(
            f"Could not locate a gene-embedding file for the pretrained model. "
            f"Checked {pretrained_dir} for 'all_embeddings*.pt', "
            f"'*gene_symbol_to_embedding*.pt', '*gene_embeddings*.pt'. "
            f"The SE-100M snapshot's config references "
            f"'{cfg.embeddings.get(cfg.embeddings.current, {}).get('all_embeddings', '?')}'. "
            f"Download the referenced file into {pretrained_dir} or adjust this locator."
        )

    # Parse architecture from the snapshot config.
    try:
        m = cfg["model"]
        arch = {
            "emsize":     int(m["emsize"]),
            "d_hid":      int(m["d_hid"]),
            "nhead":      int(m["nhead"]),
            "nlayers":    int(m["nlayers"]),
            "output_dim": int(m["output_dim"]),
            "pad_length": int(cfg.get("dataset", {}).get("pad_length", 2048)),
            "dropout":    float(m.get("dropout", DROPOUT)),
        }
    except Exception as e:
        raise RuntimeError(
            f"Could not parse model architecture from {cfg_path}: {e}. "
            f"Expected keys: model.emsize, model.d_hid, model.nhead, "
            f"model.nlayers, model.output_dim, dataset.pad_length."
        )

    print(f"  [pretrained] ckpt       : {ckpt}")
    print(f"  [pretrained] config     : {cfg_path}")
    print(f"  [pretrained] embeddings : {embeddings_path}")
    print(f"  [pretrained] arch       : {arch}")
    return {"ckpt": ckpt, "config": cfg_path, "embeddings": embeddings_path, "arch": arch}


def build_finetune_init_ckpt(orig_ckpt: Path, init_ckpt: Path) -> None:
    """Write a weights-only Lightning ckpt at ``init_ckpt``.

    Accepts either:
      * Lightning ``*.ckpt`` (wrapped dict with ``state_dict``) — strip
        trainer/optimizer state so Lightning resumes weights only.
      * HF ``*.safetensors`` (flat tensor dict) — wrap as a Lightning ckpt.

    In both cases, keys prefixed with ``dataset_`` are dropped: SE-100M was
    pretrained with ``model.dataset_correction=true`` (multi-dataset batch
    correction head), but our PBMC fine-tune sets ``dataset_correction=false``
    so the model won't have those attributes and strict state_dict loading
    would otherwise fail on unexpected keys.
    """
    import torch
    if init_ckpt.exists():
        print(f"  [init-ckpt] Stripped ckpt already at {init_ckpt}; skipping.")
        return
    print(f"  [init-ckpt] Building weights-only ckpt {init_ckpt} from {orig_ckpt.name}")

    if orig_ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(orig_ckpt))
        pl_version = "2.0.0"
    else:
        raw = torch.load(str(orig_ckpt), map_location="cpu", weights_only=False)
        state_dict = raw["state_dict"] if "state_dict" in raw else raw
        pl_version = raw.get("pytorch-lightning_version", "2.0.0")

    # SE-100M was trained with model.dataset_correction=true (z_dim_ds=10, 14418
    # dataset classes). We replicate that config during fine-tune so every
    # state_dict shape matches under Lightning's strict=True load. The downstream
    # fine-tune still only sees our PBMC manifest (2 dataset names), but the
    # classifier simply gets a meaningless label signal that doesn't crash
    # training — the encoder stack (which produces the cell embeddings we
    # actually measure MI on) gets useful gradients from the reconstruction loss.
    # No state_dict stripping needed.
    print(f"  [init-ckpt] Keeping all {len(state_dict)} keys "
          f"(fine-tune runs with dataset_correction=true to match SE-100M).")

    stripped = {
        "state_dict": state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl_version,
        # Missing keys below are re-initialised from the Hydra config at fit time.
        "optimizer_states": [],
        "lr_schedulers": [],
        # NB: we deliberately omit ``loops`` and ``callbacks``. Lightning's
        # checkpoint_connector.restore_loops() does
        # ``state_dict = self._loaded_checkpoint.get("loops")`` and skips the
        # whole block if it's None. An empty dict ``{}`` triggers a
        # ``KeyError: 'fit_loop'`` at restore_loops() line 345.
    }
    init_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, str(init_ckpt))
    print(f"  [init-ckpt] Wrote {len(state_dict)} state_dict keys -> {init_ckpt}")


# ── PBMC state_data profile keyed on the pretrained gene vocabulary ─────

def prepare_pbmc_state_profile_pretrained(
    size: int,
    quality: float,
    pretrained_embeddings: Path,
    trial_dir: Path,
) -> tuple[Path, str]:
    """Run ``state emb preprocess`` using the pretrained model's gene embeddings.

    Writes into ``trial_dir/state_data/`` so the existing PBMC profiles at
    ``data/PBMC/.../preprocessed/state_data/`` (built with our local ESM file)
    are not touched. Returns ``(profile_dir, profile_name)``.

    Re-uses the CSV manifests already produced by
    ``PrepareData.prepare_for_state`` (under ``preprocessed/state_data/``) so
    we don't have to re-tokenize or re-generate train/val csvs.
    """
    from scaling_laws.paths import STATE_PYTHON, STATE_PACKAGE_DIR, STATE_DEFAULTS_YAML

    profile_name = f"finetune_PBMC_{size}_{quality}".replace(".", "_")
    profile_dir = trial_dir / "state_data"
    marker = profile_dir / f"all_embeddings_{profile_name}.pt"
    config_path = profile_dir / f"state_config_{profile_name}.yaml"

    # Fast path: preprocess artifacts already on disk. Still patch num_datasets
    # in the cached config so it matches SE-100M (earlier versions wrote 2).
    if marker.exists():
        if config_path.exists():
            with open(config_path) as f:
                cfg_doc = yaml.safe_load(f)
            nd = cfg_doc.get("dataset", {}).get(profile_name, {}).get("num_datasets")
            if nd != 14418:
                cfg_doc["dataset"][profile_name]["num_datasets"] = 14418
                with open(config_path, "w") as f:
                    yaml.safe_dump(cfg_doc, f, sort_keys=False)
        return profile_dir, profile_name

    profile_dir.mkdir(parents=True, exist_ok=True)

    train_h5ad = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
    val_h5ad = DATA_DIR / DATASET / "validation" / str(quality) / "preprocessed" / "preprocessed.h5ad"
    test_h5ad = DATA_DIR / DATASET / "test" / str(quality) / "preprocessed" / "preprocessed.h5ad"
    for p in (train_h5ad, val_h5ad, test_h5ad):
        if not p.exists():
            raise FileNotFoundError(f"Required h5ad missing: {p}. Run Experiments.prepare_state_data() first.")

    train_csv = profile_dir / f"train_{profile_name}.csv"
    train_csv.write_text(f"species,path,names\nhuman,{train_h5ad},{profile_name}_train\n")
    combined_val_csv = profile_dir / f"val_combined_{profile_name}.csv"
    combined_val_csv.write_text(
        f"species,path,names\n"
        f"human,{val_h5ad},{profile_name}_val\n"
        f"human,{test_h5ad},{profile_name}_test\n"
    )

    # Copy defaults, then point state emb preprocess at our copy — state emb
    # preprocess mutates the file in-place to register the new profile.
    shutil.copy(STATE_DEFAULTS_YAML, config_path)

    cmd = [
        str(STATE_PYTHON), "-m", "state", "emb", "preprocess",
        "--profile-name", profile_name,
        "--train-csv", str(train_csv),
        "--val-csv", str(combined_val_csv),
        "--output-dir", str(profile_dir),
        "--config-file", str(config_path),
        "--all-embeddings", str(pretrained_embeddings),
    ]
    print(f"  [{DATASET}/{size}/{quality}] state emb preprocess (SE-100M vocab) -> {profile_dir}")
    subprocess.run(cmd, cwd=str(STATE_PACKAGE_DIR), check=True)

    # Match the val-only patching done in the existing from-scratch pipeline so
    # validation does not leak the test split. Plain PyYAML round-trip — the file
    # is a flat mapping and this avoids pulling omegaconf into the modeling env.
    with open(config_path) as f:
        cfg_doc = yaml.safe_load(f)
    preprocessed_val_csv = Path(cfg_doc["dataset"][profile_name]["val"])
    df_val = pd.read_csv(preprocessed_val_csv)
    df_val = df_val[df_val["names"].str.endswith("_val")]
    val_only_out = profile_dir / f"val_only_{profile_name}.csv"
    df_val.to_csv(val_only_out, index=False)
    cfg_doc["dataset"][profile_name]["val"] = str(val_only_out)
    # Must match SE-100M pretraining. The `dataset_encoder` classifier head has
    # shape (d_model, num_datasets); mismatching this dim makes strict ckpt load
    # fail. 14418 corresponds to the `scbasecamp-cellxgene-tahoe-filtered`
    # profile the model was pretrained on.
    cfg_doc["dataset"][profile_name]["num_datasets"] = 14418
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg_doc, f, sort_keys=False)

    assert marker.exists(), f"Expected {marker} after preprocess"
    return profile_dir, profile_name


# ── Per-(size, quality) fine-tune job (subprocess worker) ───────────────

def run_one(size: int, quality: float, device: int, pretrained: dict) -> dict:
    """Fine-tune + embed + MI for one (size, quality) cell on ``device``."""
    from scaling_laws.algo.state import State

    trial_dir = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / str(size) / str(quality)
    trial_name = f"{TRIAL_PREFIX}_{TRIAL_ID:02d}_{size}_{quality}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    arch = pretrained["arch"]

    # Grid-aware hyperparameters. Kept for lr/dropout/weight_decay only —
    # epoch count is now fixed at NUM_EPOCHS_BASE (per user request, no
    # scaling) and early stopping is disabled so each cell trains the full
    # 3 epochs regardless of dataset size.
    size_factor   = min(1.0, size / 10_000)                                  # 100k→1.0, 10k→1.0, 1k→0.1, 100→0.01
    max_lr_cell   = FINETUNE_MAX_LR * max(0.3, size_factor ** 0.5)           # √size scaling, floor at 0.3×
    max_epochs    = NUM_EPOCHS_BASE                                           # fixed 3 epochs per cell
    dropout_cell  = DROPOUT if size >= 10_000 else 0.2
    weight_decay  = 0.01 if size >= 10_000 else 0.05

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
            "num_epochs": max_epochs,
            "max_lr": max_lr_cell,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "dropout": dropout_cell,
            "weight_decay": weight_decay,
            "early_stopping": False,
            "val_check_interval": VAL_CHECK_INTERVAL,
        },
        "pretrained": {
            "hf_model_id": HF_MODEL_ID,
            "ckpt": str(pretrained["ckpt"]),
            "embeddings": str(pretrained["embeddings"]),
        },
    }
    (trial_dir / "config.json").write_text(json.dumps(cfg_doc, indent=2, default=str))

    result = {
        **{k: cfg_doc[k] for k in ("trial_id", "trial_name", "size", "quality", "seed", "device")},
        **{f"arch_{k}": v for k, v in arch.items()},
        "status": "pending",
        "train_time_s": float("nan"),
        "error": "",
    }
    t0 = time.time()

    class FinetuneState(State):
        """State with the model_path / profile_dir / checkpoint_dir rerouted
        to ``trial_dir``, and train() overridden to load pretrained weights
        before running ``state emb fit``."""

        def __init__(self, **kw):
            super().__init__(**kw)
            # Re-route artifacts under trial_dir/ (same trick as TunableState
            # in 2026-04-20_14-31_compute_state_model_sizing_pbmc.py).
            self.model_path = trial_dir
            self.model_path.mkdir(parents=True, exist_ok=True)
            # NB: save_folder_path must include the quality component — the
            # base ``State.embed()`` (scaling_laws/src/scaling_laws/algo/state.py:407)
            # writes ``save_folder_path / model_name / embeddings.npy``, and
            # when ``save_folder_path = model_path.parent`` the quality subdir
            # disappears and all 10 parallel cells race on the same file
            # (observed symptom: partial-write reshape errors mid-sweep).
            self.save_folder_path = self.model_path
            self.embeddings_path = self.model_path / "embeddings.csv"
            self.test_loss_path = self.model_path / "test_loss.txt"
            self.checkpoint_dir = self.model_path / "checkpoints"
            # Point to the SE-100M-vocab profile we built above.
            self.profile_dir = trial_dir / "state_data"
            self.config_path = self.profile_dir / f"state_config_{self.profile_name}.yaml"

        def _check_preprocessed(self) -> None:
            marker = self.profile_dir / f"all_embeddings_{self.profile_name}.pt"
            if not marker.exists():
                raise FileNotFoundError(
                    f"Pretrained-vocab preprocessing missing at {marker}. "
                    f"prepare_pbmc_state_profile_pretrained() should have been run first."
                )

        def train(self) -> None:
            import math
            import anndata as ad

            self._check_preprocessed()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for item in list(self.checkpoint_dir.iterdir()):
                shutil.rmtree(item) if item.is_dir() else item.unlink()

            # Place the stripped pretrained ckpt where STATE's trainer picks it
            # up as ``last.ckpt`` (see STATE/state/src/state/emb/utils.py:179).
            run_dir = self.checkpoint_dir / f"state_{self.profile_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(pretrained["init_ckpt"], run_dir / "last.ckpt")

            env = self._get_env()
            train_h5ad = self.train_data_path / "preprocessed.h5ad"
            num_cells = int(ad.read_h5ad(train_h5ad, backed="r").shape[0])
            batches_per_epoch = max(1, num_cells // self.batch_size)
            # Match from-scratch STATE (scaling_laws/algo/state.py:131): val + checkpoint
            # every 1000 optimizer steps. Note that at 3 epochs × 98 opt-steps/epoch ≈
            # 293 total optimizer steps, no val window fires during training — the best
            # checkpoint falls back to the end-of-training *_final.pt that Lightning
            # writes at shutdown (emb/train/trainer.py:218). Raise NUM_EPOCHS_BASE to
            # ≥11 epochs if you want at least one val window during training.
            val_interval = 1000

            print(f"  [{trial_name}] {num_cells} cells, ~{batches_per_epoch} batches/epoch, "
                  f"val every {val_interval} steps, {max_epochs} epoch(s), "
                  f"max_lr={max_lr_cell:.2e}, dropout={dropout_cell}, wd={weight_decay}")

            cmd = [
                str(self.state_python), "-m", "state", "emb", "fit",
                "--conf", str(self.config_path),
                # Profile
                f"embeddings.current={self.profile_name}",
                f"dataset.current={self.profile_name}",
                # Dataset shape
                f"dataset.num_cells={num_cells}",
                "dataset.num_train_workers=4",
                "dataset.num_val_workers=2",
                f"dataset.pad_length={self.pad_length}",
                f"dataset.P={self.pad_length // 4}",
                f"dataset.N={self.pad_length // 4}",
                f"dataset.S={self.pad_length // 4}",
                # Model arch — matches SE-100M so the stripped ckpt's state_dict fits.
                f"model.batch_size={self.batch_size}",
                f"model.emsize={self.emsize}",
                f"model.d_hid={self.d_hid}",
                f"model.nhead={self.nhead}",
                f"model.nlayers={self.nlayers}",
                f"model.output_dim={self.output_dim}",
                # Must match SE-100M pretraining (dataset_correction=true,
                # num_datasets=14418 for scbasecamp-cellxgene-tahoe-filtered).
                # With dataset_correction=false the binary_decoder is built as
                # (output_dim + d_model + 1) instead of (... + 11) and
                # Lightning's strict state_dict load rejects the ckpt. The
                # matching num_datasets=14418 is injected into the profile's
                # dataset.<profile>.num_datasets in prepare_pbmc_state_profile_pretrained.
                "model.dataset_correction=true",
                f"model.dropout={dropout_cell}",
                # Optimizer: fine-tune at 1e-6 (10× below pretrain's 1e-5 from
                # state-defaults.yaml:236). start=end=1.0 collapses LinearLR
                # warmup (emb/nn/model.py:472-477) so the schedule is pure
                # cosine decay instead of a 33%→100% ramp-up that would
                # overshoot at fine-tune scale. grad_accum=16 brings effective
                # batch to 64×16=1024 — matches SE-100M pretrain
                # (batch=128, grad_accum=8, effective=1024).
                f"optimizer.max_lr={max_lr_cell}",
                "optimizer.start=1.0",
                "optimizer.end=1.0",
                f"optimizer.gradient_accumulation_steps={GRAD_ACCUM_STEPS}",
                f"optimizer.weight_decay={weight_decay}",
                "optimizer.reset_lr_on_restart=true",
                # Experiment
                f"experiment.name=state_{self.profile_name}",
                f"experiment.num_epochs={max_epochs}",
                "experiment.num_gpus_per_node=1",
                "experiment.num_nodes=1",
                f"experiment.port={self._get_unique_port()}",
                f"experiment.val_check_interval={val_interval}",
                "+experiment.log_every_n_steps=10",
                "experiment.limit_val_batches=50",
                f"experiment.checkpoint.path={self.checkpoint_dir}",
                f"experiment.checkpoint.every_n_train_steps={val_interval}",
                "experiment.checkpoint.monitor=validation/val_loss",
                # Match from-scratch (scaling_laws/algo/state.py:193-194): keep every
                # ckpt, don't auto-write last.ckpt. The base ``State.embed()`` uses
                # ``_find_best_checkpoint()`` which argmins val_loss in metrics.csv
                # and falls back to the end-of-training *_final.pt when no val window
                # fires (relevant here because 3 epochs × 98 opt-steps = 293 < 1000).
                "experiment.checkpoint.save_top_k=-1",
                "+experiment.checkpoint.save_last=false",
                # Disable external loggers + extra validations
                "wandb.enable=false",
                "validations.diff_exp.enable=false",
                "validations.perturbation.enable=false",
                # Early stopping DISABLED — run the full fixed 3-epoch budget.
                # Previous sweep observed StepBasedEarlyStopping firing after
                # ~1.28 epochs (step 500 with defaults 500/3) which cut training
                # short. Disable it entirely so max_epochs is the only cap.
                "experiment.early_stopping.enable=false",
            ]
            try:
                subprocess.run(cmd, cwd=str(self.state_package_dir), env=env, check=True)
            finally:
                self._save_loss_curves()
                self._save_final_losses()

    try:
        model = FinetuneState(
            base_dir=str(DATA_DIR / DATASET / str(size) / str(quality)),
            device=device,
            dataset_name=DATASET,
            seed=SEED,
            pad_length=arch["pad_length"],
            emsize=arch["emsize"],
            d_hid=arch["d_hid"],
            nhead=arch["nhead"],
            nlayers=arch["nlayers"],
            output_dim=arch["output_dim"],
            batch_size=BATCH_SIZE,
            max_lr=max_lr_cell,
            dropout=dropout_cell,
        )
        # Override the profile_name so train() / _check_preprocessed see the
        # SE-100M profile we built in the parent process.
        model.profile_name = f"finetune_PBMC_{size}_{quality}".replace(".", "_")
        model.profile_dir = trial_dir / "state_data"
        model.config_path = model.profile_dir / f"state_config_{model.profile_name}.yaml"

        t = time.time(); print(f"[{trial_name}] train START GPU={device}", flush=True)
        model.train()
        print(f"[{trial_name}] train DONE t={time.time()-t:.0f}s", flush=True)

        t = time.time(); print(f"[{trial_name}] embed START", flush=True)
        model.embed()
        print(f"[{trial_name}] embed DONE t={time.time()-t:.0f}s", flush=True)

        t = time.time(); print(f"[{trial_name}] MI    START", flush=True)
        mi = model.mutual_information()
        for sig, val in mi.items():
            result[f"mi_{sig.replace('.csv','')}"] = float(val)
        print(f"[{trial_name}] MI    DONE t={time.time()-t:.0f}s", flush=True)

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


# ── Orchestrator ────────────────────────────────────────────────────────

def detect_gpus() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pretrained model: download + parse.
    download_pretrained(PRETRAINED_DIR)
    pretrained = locate_pretrained(PRETRAINED_DIR)
    init_ckpt = PRETRAINED_DIR / "finetune_init.ckpt"
    build_finetune_init_ckpt(pretrained["ckpt"], init_ckpt)
    pretrained["init_ckpt"] = init_ckpt

    # 2. Verify the PBMC noise-scaling grid is ready on disk.
    missing = []
    for size in SIZES:
        for quality in QUALITIES:
            for split in ("preprocessed", "validation", "test"):
                if split == "preprocessed":
                    p = DATA_DIR / DATASET / str(size) / str(quality) / "preprocessed" / "preprocessed.h5ad"
                else:
                    p = DATA_DIR / DATASET / split / str(quality) / "preprocessed" / "preprocessed.h5ad"
                if not p.exists():
                    missing.append(p)
    if missing:
        print(f"MISSING {len(missing)} h5ads — run Experiments.prepare_state_data() first. First 10:")
        for p in missing[:10]:
            print(f"  {p}")
        sys.exit(1)
    print(f"All {len(SIZES)} x {len(QUALITIES)} (size, quality) h5ads verified.")

    # 3. Rebuild the state_data profile at each (size, quality) using the
    # pretrained model's gene vocabulary. Each cell writes into its own
    # trial_dir/state_data/ so parallel builds don't collide; the heavy lifting
    # is a subprocess call to `state emb preprocess` which releases the GIL.
    profile_jobs = [(s, q) for s in SIZES for q in QUALITIES]
    profile_workers = min(len(profile_jobs), (os.cpu_count() or 4))
    print(f"\nBuilding SE-100M-vocab state_data profiles (parallel, workers={profile_workers})...")

    def _build_profile(sq):
        s, q = sq
        td = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / str(s) / str(q)
        td.mkdir(parents=True, exist_ok=True)
        prepare_pbmc_state_profile_pretrained(s, q, pretrained["embeddings"], td)

    with ThreadPoolExecutor(max_workers=profile_workers) as pex:
        futs = [pex.submit(_build_profile, sq) for sq in profile_jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="profiles", unit="cell"):
            fut.result()  # re-raise any error

    # 4. Schedule the fine-tune grid across GPUs.
    gpus = detect_gpus()
    slots = gpus * JOBS_PER_GPU
    max_workers = max(1, len(slots))
    print(f"\nGPUs={gpus} JOBS_PER_GPU={JOBS_PER_GPU} slots={max_workers}")

    jobs = [(size, quality) for size in SIZES for quality in QUALITIES]
    # Skip cells that already completed (resume-friendly).
    pending = []
    for size, quality in jobs:
        trial_dir = OUTPUT_DIR / f"{TRIAL_PREFIX}_{TRIAL_ID:02d}" / str(size) / str(quality)
        res = trial_dir / "result.json"
        if res.exists():
            try:
                status = json.loads(res.read_text()).get("status")
                if status == "ok":
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
    pbar = tqdm(total=total, desc="finetune SE-100M", unit="cell", dynamic_ncols=True)
    n_ok = n_err = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while pending and free_slots:
            size, quality = pending.pop(0)
            device = free_slots.pop(0)
            fut = ex.submit(run_one, size, quality, device, pretrained)
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
                fut = ex.submit(run_one, size, quality, device, pretrained)
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
    for size_dir in sorted(p for p in trial_root.iterdir() if p.is_dir()):
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

    # Per-trial YAML (mirrors the model_sizing layout).
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
