"""STATE hyperparameter sweep on PBMC (size x quality grid).

Architecture is held fixed at STATE defaults (emsize=256, d_hid=512, nhead=4,
nlayers=3, output_dim=256). Only regularization/optimization knobs are swept.
(N_TRIALS + 1) x |SIZES| x |QUALITIES| runs — one extra "production" trial is
always appended with the exact config used everywhere else in this project
(scaling_laws.algo.state.State defaults; see PRODUCTION_CONFIG below), so the
sweep produces a baseline curve directly comparable to the all-datasets run.

Tunable parameters (sweep):
    dropout, batch_size, max_lr, weight_decay

Fixed architecture (FIXED_ARCH):
    emsize, d_hid, nhead, nlayers, output_dim, pad_length

Training protocol — matches scaling_laws/algo/state.py used by the
all-datasets run (analysis/2026-04-16_14-49_compute_state_all_datasets.py):
    - step budget: OPTIMIZER_STEPS=15000, enforced via experiment.num_epochs =
      ceil(OPTIMIZER_STEPS / batches_per_epoch) — NOT the profiler path
    - early stopping ALWAYS ON, monitor=validation/val_loss, patience=5,
      every_n_steps=VAL_CHECK_INTERVAL
    - validation every VAL_CHECK_INTERVAL=1000 optimizer steps (limit_val_batches=50)
    - train loss logged every 10 optimizer steps
    - one .ckpt saved per validation tick (save_top_k=-1) so the best ckpt can
      be recovered via argmin(val_loss) in metrics.csv
    - after training, full Lightning metrics.csv is copied to
      <trial>/loss/metrics.csv and best train/val/test scalars are written to
      train_loss.txt / val_loss.txt / test_loss.txt next to it

Parallelism: 2 jobs per GPU across all visible GPUs.

Outputs:
    $NOISE_SCALING_OUTPUT_BASE/hp_tunning/  (default ~/noise_scaling/data/other/hp_tunning)
        sweep_results.csv       # full results table
        hp_trial_XX.yaml        # per-config YAML with params + per-size results
"""

from __future__ import annotations

import atexit
import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DATASET = "PBMC"
SIZES = [100000]  # largest size only
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
from scaling_laws.paths import DATA_DIR, OUTPUT_BASE
# All per-trial artifacts (checkpoints, embeddings, loss curves, MI results)
# live under OUTPUT_DIR/<trial_name>/, independent of DATA_DIR.
OUTPUT_DIR = OUTPUT_BASE / "hp_tunning"

N_TRIALS = 8  # 8 random HP draws + 1 appended PRODUCTION_CONFIG trial = 9 trials total
# Prefix for trial folder/yaml/log names. Change e.g. to "model_sizing" to
# generate `model_sizing_00/sz100000/qQUAL/` and `model_sizing_00.yaml`.
TRIAL_PREFIX = "hp_trial"
JOBS_PER_GPU = 3  # 3/GPU on 80GB H100s (reduced from 4: batch_size=128 trials were OOMing)
SEED = 42
# Hard cap on total optimizer (gradient) steps per trial. Wired via STATE's
# profiler path so Lightning's trainer.max_steps is honored. Matches the
# 15k-step budget used by the production all-datasets run
# (analysis/2026-04-16_14-49_compute_state_all_datasets.py) so HP-sweep loss
# curves are directly comparable.
OPTIMIZER_STEPS = 15000
# Validation cadence (in optimizer steps). Matches scaling_laws/algo/state.py
# so train/val curves and best-ckpt selection work the same way as the
# all-datasets run.
VAL_CHECK_INTERVAL = 1000


# ── Search spaces ─────────────────────────────────────────────────────
# Every parameter from state-defaults.yaml model/task/optimizer sections.
# Format: (sampling_kind, *args)

# Architecture held fixed at STATE defaults (see scaling_laws algo/state.py).
FIXED_ARCH = {
    "emsize":     256,
    "d_hid":      512,
    "nhead":      4,
    "nlayers":    3,
    "output_dim": 256,
    "pad_length": 2048,
}

SEARCH_SPACE = {
    # Regularization
    "dropout":      ("choice", [0.0, 0.1, 0.2, 0.3]),
    "batch_size":   ("choice", [32, 64, 128]),
    # Optimizer
    "max_lr":       ("choice", [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
    "weight_decay": ("choice", [1e-4, 1e-3, 1e-2, 1e-1]),
}

# Production baseline: the exact config used everywhere else in this repo
# (scaling_laws.algo.state.State defaults + hardcoded weight_decay from the
# `state emb fit` command in that module). Always appended as the final trial
# so the sweep always includes a curve directly comparable to the all-datasets
# run (analysis/2026-04-16_14-49_compute_state_all_datasets.py).
PRODUCTION_CONFIG = {
    "dropout":      0.1,
    "batch_size":   64,
    "max_lr":       1e-4,
    "weight_decay": 1e-2,
}


def generate_trials(n: int, seed: int) -> list[dict]:
    """Sample n distinct HP configs from SEARCH_SPACE, then append PRODUCTION_CONFIG.

    Returns n + 1 trials: `n` random draws without replacement (production
    excluded from the sampling pool so it can't be drawn twice), plus the
    hardcoded production baseline as trial_id = n.
    """
    names = list(SEARCH_SPACE)
    for name, spec in SEARCH_SPACE.items():
        if spec[0] != "choice":
            raise ValueError(f"{name}: unsupported sampling kind {spec[0]!r}")
    prod_tuple = tuple(PRODUCTION_CONFIG[name] for name in names)
    grid = [p for p in itertools.product(*(SEARCH_SPACE[name][1] for name in names))
            if p != prod_tuple]
    if n > len(grid):
        raise ValueError(f"N_TRIALS={n} exceeds grid size {len(grid)} (production excluded)")
    picks = random.Random(seed).sample(grid, n)
    trials = [
        {"trial_id": i, **dict(zip(names, pick))}
        for i, pick in enumerate(picks)
    ]
    trials.append({"trial_id": n, **PRODUCTION_CONFIG})
    return trials


# ── Per-trial runner (executed in a subprocess) ─────────────────────────

def run_trial(trial: dict, device: int) -> dict:
    """Train STATE with `trial` hyperparameters on `device`.

    The trial dict must contain 'size' and 'quality' keys; base_dir is
    derived from them. Total gradient steps are hard-capped by OPTIMIZER_STEPS.

    Runs in its own subprocess so that the worker GPU and env are isolated.
    Each trial writes into its own OUTPUT_DIR/<trial_name>/ directory.

    Returns dict with trial config + metrics (best_val_loss, etc.).
    """
    # Imports are done inside the worker to avoid CUDA init in the parent.
    # `scaling_laws` is pip-installed (editable) in the env that launches this
    # script; the ProcessPoolExecutor worker inherits sys.path via fork, so no
    # manual path insertion is needed.
    from scaling_laws.algo.state import State

    trial_name = trial["trial_name"]
    subpath = trial["subpath"]
    size = trial["size"]
    quality = trial["quality"]
    base_dir = DATA_DIR / DATASET / str(size) / str(quality)

    class TunableState(State):
        """State variant that exposes ALL model/task/optimizer params as
        Hydra overrides.  Each trial writes to its own results/State/<model_name>/."""

        def __init__(self, model_name: str = "model", subpath: str | None = None,
                     extra_hparams: dict | None = None, **kw):
            super().__init__(**kw)
            # Re-route every artifact to OUTPUT_DIR/<subpath>/ so models live
            # in nested folders (hp_trial_NN/szSIZE/qQUALITY/), not under
            # DATA_DIR/PBMC/.../results. `model_name` stays flat so it can be
            # used as Lightning's experiment.name / log subdir.
            #
            # IMPORTANT: BaseAlgorithm.mutual_information builds its output
            # path as `self.save_folder_path / self.model_path.name / "MI" / ...`,
            # which assumes model_path is one level under save_folder_path.
            # We preserve that invariant by setting save_folder_path to the
            # PARENT of the nested model_path; that way MI lands at
            # OUTPUT_DIR/<subpath>/MI/<seed>/<signal>/ (correct), not at
            # OUTPUT_DIR/<last-segment>/MI/... (which would flatten + collide).
            self.model_name = model_name
            self.subpath = subpath if subpath is not None else model_name
            self.model_path = OUTPUT_DIR / self.subpath
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.save_folder_path = self.model_path.parent
            self.embeddings_path = self.model_path / "embeddings.csv"
            self.test_loss_path = self.model_path / "test_loss.txt"
            self.checkpoint_dir = self.model_path / "checkpoints"
            # Extra hparams not covered by State's constructor
            self.hp = extra_hparams or {}

        def _bool(self, v) -> str:
            return "true" if v else "false"

        def train(self) -> None:
            import math
            import anndata as ad
            self._check_preprocessed()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for item in self.checkpoint_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()

            env = self._get_env()
            train_h5ad = self.train_data_path / "preprocessed.h5ad"
            adata = ad.read_h5ad(train_h5ad, backed="r")
            num_cells = int(adata.shape[0])
            batches_per_epoch = max(1, num_cells // self.batch_size)
            val_interval = VAL_CHECK_INTERVAL
            max_epochs = max(1, math.ceil(OPTIMIZER_STEPS / batches_per_epoch))
            print(f"  [{self.model_name}] {num_cells} cells, ~{batches_per_epoch} batches/epoch, "
                  f"val every {val_interval} steps, step budget {OPTIMIZER_STEPS} -> {max_epochs} epochs")

            # Mirrors scaling_laws/algo/state.py:144-211 exactly. Swept HPs
            # (batch_size, dropout, max_lr, weight_decay) + fixed arch are
            # injected via the same Hydra keys; the step cap is enforced by
            # converting to num_epochs (NOT the profiler path), and early
            # stopping is ALWAYS on with patience=5, just like production.
            hp = self.hp
            cmd_fit = [
                str(self.state_python), "-m", "state", "emb", "fit",
                "--conf", str(self.config_path),
                # Profile selection
                f"embeddings.current={self.profile_name}",
                f"dataset.current={self.profile_name}",
                # Dataset
                f"dataset.num_cells={num_cells}",
                "dataset.num_train_workers=4",
                "dataset.num_val_workers=2",
                f"dataset.pad_length={self.pad_length}",
                f"dataset.P={self.pad_length // 4}",
                f"dataset.N={self.pad_length // 4}",
                f"dataset.S={self.pad_length // 4}",
                # Model architecture (swept HPs: batch_size, dropout)
                f"model.batch_size={self.batch_size}",
                f"model.emsize={self.emsize}",
                f"model.d_hid={self.d_hid}",
                f"model.nhead={self.nhead}",
                f"model.nlayers={self.nlayers}",
                f"model.output_dim={self.output_dim}",
                "model.dataset_correction=false",
                f"model.dropout={self.dropout}",
                # Optimizer (swept HPs: max_lr, weight_decay)
                f"optimizer.max_lr={self.max_lr}",
                "optimizer.gradient_accumulation_steps=1",
                f"optimizer.weight_decay={hp.get('weight_decay', 0.01)}",
                # Experiment — name MUST stay 'state_{profile_name}' so the
                # inherited _find_lightning_metrics_csv can locate metrics.csv.
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
                "experiment.checkpoint.save_top_k=-1",
                "+experiment.checkpoint.save_last=false",
                # Disable logging and extra validations
                "wandb.enable=false",
                "validations.diff_exp.enable=false",
                "validations.perturbation.enable=false",
                # Early stopping ALWAYS ON (patience=5) — matches production
                "experiment.early_stopping.enable=true",
                "experiment.early_stopping.monitor=validation/val_loss",
                "experiment.early_stopping.patience=5",
                f"+experiment.early_stopping.every_n_steps={val_interval}",
                "experiment.early_stopping.min_delta=0.0",
                "experiment.early_stopping.mode=min",
            ]
            try:
                subprocess.run(cmd_fit, cwd=str(self.state_package_dir), env=env, check=True)
            finally:
                # Persist full Lightning loss curve + scalar train/val/test files
                # even on crash / SIGTERM (mirrors scaling_laws/algo/state.py).
                self._save_loss_curves()
                self._save_final_losses()

        # The base implementations write to self.save_folder_path/self.model_name/...
        # but in this sweep model_name == trial_name (a flat label) and that
        # path doesn't equal self.model_path. Override to write into
        # self.model_path directly so the loss artifacts land alongside
        # config.json / embeddings.csv / MI/.
        def _save_loss_curves(self) -> None:
            metrics_file = self._find_lightning_metrics_csv()
            if metrics_file is None:
                print(f"  Warning: no Lightning metrics.csv under "
                      f"{self.checkpoint_dir}/state_{self.profile_name}/version_*; "
                      f"train/val loss curves will NOT be saved.")
                return
            loss_dir = self.model_path / "loss"
            loss_dir.mkdir(parents=True, exist_ok=True)
            dest = loss_dir / "metrics.csv"
            shutil.copy(metrics_file, dest)
            try:
                df = pd.read_csv(dest)
                n_train = df["trainer/train_loss"].notna().sum() if "trainer/train_loss" in df.columns else 0
                n_val = df["validation/val_loss"].notna().sum() if "validation/val_loss" in df.columns else 0
                last_step = int(df["step"].max()) if "step" in df.columns and not df.empty else 0
                print(f"  Loss curves saved to {dest} "
                      f"(train_pts={n_train}, val_pts={n_val}, last_step={last_step})")
            except Exception as e:
                print(f"  Loss curves saved to {dest} (could not summarize: {e})")

        def _save_final_losses(self) -> None:
            metrics_path = self.model_path / "loss" / "metrics.csv"
            if not metrics_path.exists():
                print(f"  Warning: {metrics_path} missing — final losses NOT saved.")
                return
            try:
                df = pd.read_csv(metrics_path, on_bad_lines="skip")
            except Exception as e:
                print(f"  Warning: could not read {metrics_path}: {e}")
                return
            train_loss_path = self.model_path / "train_loss.txt"
            val_loss_path = self.model_path / "val_loss.txt"
            def _best(col: str):
                if col not in df.columns:
                    return None
                vals = df[col].dropna()
                return float(vals.min()) if not vals.empty else None
            best_train = _best("trainer/train_loss")
            best_val = _best("validation/val_loss")
            if best_train is not None:
                train_loss_path.write_text(f"{best_train:.6f}")
                print(f"  Final train loss: {best_train:.6f} -> {train_loss_path}")
            if best_val is not None:
                val_loss_path.write_text(f"{best_val:.6f}")
                self.test_loss_path.write_text(f"{best_val:.6f}")
                print(f"  Final val loss  : {best_val:.6f} -> {val_loss_path}")
                print(f"  Final test loss : {best_val:.6f} -> {self.test_loss_path}  (proxy = best val_loss)")

    # Persist the full per-trial config to its own JSON in the trial folder,
    # before training starts (so it survives crashes).
    trial_dir = OUTPUT_DIR / subpath
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_doc = {
        "trial_id": int(trial["trial_id"]),
        "trial_name": trial_name,
        "subpath": subpath,
        "dataset": DATASET,
        "size": int(size),
        "quality": float(quality),
        "device": int(device),
        "seed": SEED,
        "optimizer_steps": OPTIMIZER_STEPS,
        "swept_params": {k: trial[k] for k in SEARCH_SPACE if k in trial},
        "fixed_arch": FIXED_ARCH,
    }
    config_json_path = trial_dir / "config.json"
    config_json_path.write_text(json.dumps(config_doc, indent=2, default=str))
    print(f"[{trial_name}] config -> {config_json_path}", flush=True)

    t0 = time.time()
    result = {
        **trial,
        "device": device,
        "status": "pending",
        "train_time_s": float("nan"),
        "error": "",
    }

    try:
        # Separate params into State constructor args vs extra Hydra overrides
        extra_hparams = {
            k: trial[k] for k in trial
            if k not in ("trial_id", "trial_name", "subpath", "size", "quality",
                         "dropout", "batch_size", "max_lr")
        }
        print(f"[{trial_name}] phase=construct  base_dir={base_dir}", flush=True)
        model = TunableState(
            base_dir=str(base_dir),
            device=device,
            dataset_name=DATASET,
            seed=SEED,
            model_name=trial_name,
            subpath=subpath,
            max_lr=trial["max_lr"],
            dropout=trial["dropout"],
            batch_size=trial["batch_size"],
            extra_hparams=extra_hparams,
            **FIXED_ARCH,
        )

        t_phase = time.time()
        print(f"[{trial_name}] phase=train START  GPU={device}  out={model.model_path}", flush=True)
        model.train()
        print(f"[{trial_name}] phase=train DONE   t={time.time()-t_phase:.0f}s", flush=True)

        t_phase = time.time()
        print(f"[{trial_name}] phase=embed START  -> {model.embeddings_path}", flush=True)
        model.embed()
        print(f"[{trial_name}] phase=embed DONE   t={time.time()-t_phase:.0f}s", flush=True)

        t_phase = time.time()
        print(f"[{trial_name}] phase=mi    START  out={model.model_path / 'MI'}", flush=True)
        mi_results = model.mutual_information()
        for signal_name, mi_val in mi_results.items():
            col = signal_name.replace(".csv", "")
            result[f"mi_{col}"] = float(mi_val)
            print(f"[{trial_name}]   mi[{col}] = {mi_val:.4f}", flush=True)
        print(f"[{trial_name}] phase=mi    DONE   t={time.time()-t_phase:.0f}s", flush=True)

        result["status"] = "ok"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[{trial_name}] FAILED on GPU {device}: {result['error']}", flush=True)
    finally:
        result["train_time_s"] = time.time() - t0
        # Persist the result alongside config.json so each trial folder is
        # self-describing (config + outcome) without needing the global CSV.
        result_doc = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in result.items()}
        (trial_dir / "result.json").write_text(json.dumps(result_doc, indent=2, default=str))
        print(f"[{trial_name}] result -> {trial_dir / 'result.json'}  status={result['status']}", flush=True)

    return result


# ── Orchestrator ────────────────────────────────────────────────────────

def detect_gpus() -> list[int]:
    """Detect visible GPU indices via nvidia-smi (mirrors Experiments._detect_gpus)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def main():
    # Verify preprocessed profiles exist for all size x quality combos
    for size in SIZES:
        for quality in QUALITIES:
            quality_str = str(quality)
            base_dir = DATA_DIR / DATASET / str(size) / quality_str
            assert base_dir.exists(), f"Base dir missing: {base_dir}"
            profile_name = f"scaling_PBMC_{size}_{quality_str}".replace(".", "_")
            profile = base_dir / "preprocessed" / "state_data" / f"all_embeddings_{profile_name}.pt"
            assert profile.exists(), f"STATE profile missing: {profile}. Run prepare_state_data() first."
    print(f"All {len(SIZES)} sizes x {len(QUALITIES)} qualities verified.")

    gpus = detect_gpus()
    slots = gpus * JOBS_PER_GPU
    max_workers = len(slots)
    print(f"GPUs={gpus}  jobs_per_gpu={JOBS_PER_GPU}  slots={max_workers}")

    hp_configs = generate_trials(N_TRIALS, seed=SEED)
    print(f"Generated {len(hp_configs)} HP configs:")
    for t in hp_configs:
        print(f"  {TRIAL_PREFIX}_{t['trial_id']:02d}: " + json.dumps({k: v for k, v in t.items() if k != 'trial_id'}))

    # Cross product: each HP config x each size x each quality.
    # `trial_name` is flat (used as Lightning's experiment.name and in logs);
    # `subpath` is the nested filesystem layout (<TRIAL_PREFIX>_NN/<size>/<quality>).
    jobs = []
    for cfg in hp_configs:
        for size in SIZES:
            for quality in QUALITIES:
                job = {**cfg}
                job["size"] = size
                job["quality"] = quality
                tid_tag = f"{TRIAL_PREFIX}_{cfg['trial_id']:02d}"
                sz_tag = str(size)         # e.g. "100000"
                q_tag = str(quality)       # e.g. "0.1072766"
                job["trial_name"] = f"{tid_tag}_{sz_tag}_{q_tag}"
                job["subpath"] = f"{tid_tag}/{sz_tag}/{q_tag}"
                jobs.append(job)
    random.Random(SEED).shuffle(jobs)

    total_jobs = len(jobs)
    print(f"\nTotal jobs: {len(hp_configs)} configs x {len(SIZES)} sizes x "
          f"{len(QUALITIES)} qualities = {total_jobs}  "
          f"(optimizer_steps={OPTIMIZER_STEPS} per trial)")
    # Per-trial config.json is written by run_trial into each trial's folder.

    # Save results to dedicated output dir
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    free_slots = list(slots)
    results: list[dict] = []
    pending = list(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}

        # Seed workers
        while pending and free_slots:
            trial = pending.pop(0)
            device = free_slots.pop(0)
            fut = ex.submit(run_trial, trial, device)
            futures[fut] = (trial, device)
            print(f"Submitted {trial['trial_name']} on GPU {device}")

        pbar = tqdm(total=total_jobs, desc="trials", unit="trial",
                    dynamic_ncols=True, smoothing=0.1)
        n_ok = n_err = 0
        while futures:
            done_fut = next(as_completed(futures))
            trial, device = futures.pop(done_fut)
            try:
                res = done_fut.result()
            except Exception as e:
                res = {**trial, "device": device, "status": "error",
                       "train_time_s": float("nan"),
                       "error": f"{type(e).__name__}: {e}"}
            if res["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
            mi_str = " ".join(f"{k}={v:.3f}" for k, v in res.items()
                              if k.startswith("mi_") and not np.isnan(v))
            pbar.set_postfix(ok=n_ok, err=n_err, last=res['trial_name'][:32], refresh=False)
            pbar.update(1)
            pbar.write(f"  Finished {res['trial_name']} status={res['status']} "
                       f"{mi_str} t={res['train_time_s']:.0f}s")
            results.append(res)
            free_slots.append(device)

            if pending and free_slots:
                trial = pending.pop(0)
                device = free_slots.pop(0)
                fut = ex.submit(run_trial, trial, device)
                futures[fut] = (trial, device)
                pbar.write(f"  Submitted {trial['trial_name']} on GPU {device}")
        pbar.close()

    df = pd.DataFrame(results).sort_values(["trial_id", "size", "quality"]).reset_index(drop=True)

    # Save full results CSV
    out_csv = output_dir / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")

    # Save each HP config as a YAML file with its per-size val losses
    for trial_id in sorted(df["trial_id"].unique()):
        sub = df[df["trial_id"] == trial_id]
        row0 = sub.iloc[0]
        # Build config from all sweep params
        config = {"trial_id": int(trial_id), "pad_length": 2048}
        for param_name in SEARCH_SPACE:
            if param_name in row0.index:
                val = row0[param_name]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, (np.bool_,)):
                    val = bool(val)
                config[param_name] = val
        # Detect MI columns dynamically from the dataframe
        mi_cols = [c for c in sub.columns if c.startswith("mi_")]

        per_size_quality: dict = {}
        for _, r in sub.iterrows():
            entry = {
                "optimizer_steps": OPTIMIZER_STEPS,
                "status": r["status"],
                "train_time_s": round(float(r["train_time_s"]), 1),
            }
            if mi_cols:
                entry["mi"] = {
                    c.removeprefix("mi_"): round(float(r[c]), 5)
                    for c in mi_cols if pd.notna(r.get(c))
                }
            size_key = int(r["size"])
            quality_key = float(r["quality"])
            per_size_quality.setdefault(size_key, {})[quality_key] = entry
        trial_doc = {"config": config, "results_per_size_quality": per_size_quality}
        yaml_path = output_dir / f"{TRIAL_PREFIX}_{int(trial_id):02d}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(trial_doc, f, default_flow_style=False, sort_keys=False)
    print(f"Per-trial YAML configs -> {output_dir}/{TRIAL_PREFIX}_*.yaml")

    # Print summary: mean MI across sizes x qualities per trial
    ok_df = df[df["status"] == "ok"]
    mi_cols = [c for c in ok_df.columns if c.startswith("mi_")]
    if not mi_cols:
        print("\nNo MI columns to summarize.")
        return
    agg_cols = {c: "mean" for c in mi_cols}
    primary_mi = mi_cols[0]
    summary = ok_df.groupby("trial_id").agg(agg_cols).sort_values(primary_mi, ascending=False)

    print(f"\nMean MI across sizes x qualities (sorted by {primary_mi}, higher=better):")
    for trial_id, row in summary.iterrows():
        row0 = df[df["trial_id"] == trial_id].iloc[0]
        mi_str = " ".join(f"{c}={row[c]:.3f}" for c in mi_cols if pd.notna(row[c]))
        print(f"  trial_{int(trial_id):02d}: {mi_str}  "
              f"(lr={row0['max_lr']:.2e} bs={int(row0['batch_size'])} "
              f"do={row0['dropout']:.2f} wd={row0['weight_decay']:.2e})")


if __name__ == "__main__":
    main()
