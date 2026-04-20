"""STATE model-size sweep on PBMC (model_size x quality grid).

Goal: study how STATE's training behavior scales with parameter count, holding
data fixed at the largest available PBMC size and varying only architecture.
We sweep 5 model configs spanning roughly 5M -> 100M transformer-body params
(STATE's ~109M pe_embedding sits on top of all of them and is not scaled).
For each config we run all qualities on the largest dataset size.

Architecture ratios are kept constant across configs:
    d_hid       = 2 * emsize           (FFN hidden = 2 x model dim)
    nhead       = emsize / 64          (head dim fixed at 64)
    output_dim  = emsize               (projector dim tracks model dim)
    pad_length  = 2048                 (constant — input length is data-driven)

Only `nlayers` is chosen by hand to span the desired parameter range.

Hyperparameters held fixed at STATE defaults across all configs (see
state-defaults.yaml): max_lr, dropout, batch_size, weight_decay.

Loss logging matches the HP-sweep / all-datasets runs:
    - validation every VAL_CHECK_INTERVAL=1000 optimizer steps (limit_val_batches=50)
    - train loss logged every 10 optimizer steps
    - one .ckpt per validation tick (save_top_k=-1) -> argmin(val_loss) selects best
    - after training, full Lightning metrics.csv is copied to
      <trial>/<model_name>/loss/metrics.csv and best train/val/test scalars are
      written to train_loss.txt / val_loss.txt / test_loss.txt

Parallelism: 1 job per GPU across all visible GPUs (the larger configs need
the headroom).

Outputs:
    /home/igor/noise_scaling/data/other/model_sizing/
        sweep_results.csv             # full results table
        model_sizing_NN.yaml          # per-config YAML with arch + per-quality results
        model_sizing_NN/<size>/<quality>/
            config.json
            result.json
            checkpoints/.../metrics.csv
            MI/<seed>/Y_<signal>_<quality>/lmi_mutual_information.txt
"""

from __future__ import annotations

import atexit
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
SIZES = [100000]  # largest PBMC size only (the variable here is model size)
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
DATA_DIR = Path("/home/igor/noise_scaling/data")
OUTPUT_DIR = Path("/home/igor/noise_scaling/data/other/model_sizing")

TRIAL_PREFIX = "model_sizing"
JOBS_PER_GPU = 1  # 1/GPU on 40GB A100s — the larger configs need full headroom
SEED = 42
# Hard cap on total optimizer (gradient) steps per trial — same budget as the
# HP sweep / all-datasets runs so loss curves are directly comparable.
OPTIMIZER_STEPS = 15000
VAL_CHECK_INTERVAL = 1000


# ── Hyperparameters held fixed across all model sizes (STATE defaults) ─────
FIXED_HPARAMS = {
    "max_lr":       5e-4,
    "dropout":      0.1,
    "batch_size":   64,
    "weight_decay": 0.01,
}


# ── Model-size configurations ────────────────────────────────────────────
# 5 configs spanning ~5M -> ~100M transformer-body params. Architecture ratios
# are held constant: d_hid = 2*emsize, nhead = emsize/64, output_dim = emsize.
# Only emsize and nlayers vary. pad_length stays at the data-driven default.
#
# Approximate transformer-body param counts (per layer ~ 12 * emsize^2 with
# d_hid = 2*emsize), excluding STATE's ~109M pe_embedding which is shared:
#   trial 0 — emsize=192, nlayers=4   ~  1.8M   (smallest)
#   trial 1 — emsize=320, nlayers=4   ~  4.9M
#   trial 2 — emsize=448, nlayers=6   ~ 14.4M
#   trial 3 — emsize=640, nlayers=8   ~ 39.3M
#   trial 4 — emsize=896, nlayers=10  ~ 96.4M   (largest)
MODEL_CONFIGS = [
    {"emsize": 192, "d_hid":  384, "nhead":  3, "nlayers":  4, "output_dim": 192, "pad_length": 2048},
    {"emsize": 320, "d_hid":  640, "nhead":  5, "nlayers":  4, "output_dim": 320, "pad_length": 2048},
    {"emsize": 448, "d_hid":  896, "nhead":  7, "nlayers":  6, "output_dim": 448, "pad_length": 2048},
    {"emsize": 640, "d_hid": 1280, "nhead": 10, "nlayers":  8, "output_dim": 640, "pad_length": 2048},
    {"emsize": 896, "d_hid": 1792, "nhead": 14, "nlayers": 10, "output_dim": 896, "pad_length": 2048},
]


def generate_trials() -> list[dict]:
    """Wrap MODEL_CONFIGS into trial dicts with sequential trial_ids."""
    return [{"trial_id": i, **cfg} for i, cfg in enumerate(MODEL_CONFIGS)]


# ── Per-trial runner (executed in a subprocess) ─────────────────────────

def run_trial(trial: dict, device: int) -> dict:
    """Train STATE with `trial`'s architecture on `device`.

    The trial dict must contain 'size' and 'quality' keys plus the architecture
    fields (emsize, d_hid, nhead, nlayers, output_dim, pad_length). Hyperparameters
    come from FIXED_HPARAMS (constant across the sweep).
    """
    sys.path.insert(
        0,
        "/home/igor/noise_scaling/modeling/Scaling-up-measurement-noise-scaling-laws/scaling_laws/src",
    )
    from scaling_laws.algo.state import State

    trial_name = trial["trial_name"]
    subpath = trial["subpath"]
    size = trial["size"]
    quality = trial["quality"]
    base_dir = DATA_DIR / DATASET / str(size) / str(quality)

    class TunableState(State):
        """State variant that exposes ALL model/task/optimizer params as
        Hydra overrides. Each trial writes to its own results/State/<model_name>/."""

        def __init__(self, model_name: str = "model", subpath: str | None = None,
                     extra_hparams: dict | None = None, **kw):
            super().__init__(**kw)
            # Re-route every artifact to OUTPUT_DIR/<subpath>/. See HP-sweep
            # script for the rationale on save_folder_path = model_path.parent
            # (preserves BaseAlgorithm.mutual_information's path invariant).
            self.model_name = model_name
            self.subpath = subpath if subpath is not None else model_name
            self.model_path = OUTPUT_DIR / self.subpath
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.save_folder_path = self.model_path.parent
            self.embeddings_path = self.model_path / "embeddings.csv"
            self.test_loss_path = self.model_path / "test_loss.txt"
            self.checkpoint_dir = self.model_path / "checkpoints"
            self.hp = extra_hparams or {}

        def train(self) -> None:
            import anndata as ad
            self._check_preprocessed()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for item in self.checkpoint_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()

            env = self._get_env()
            train_h5ad = self.train_data_path / "preprocessed.h5ad"
            adata = ad.read_h5ad(train_h5ad, backed="r")
            num_cells = int(adata.shape[0])
            print(f"  [{self.model_name}] {num_cells} cells, cap={OPTIMIZER_STEPS} optimizer steps")

            hp = self.hp
            cmd_fit = [
                str(self.state_python), "-m", "state", "emb", "fit",
                "--conf", str(self.config_path),
                f"embeddings.current={self.profile_name}",
                f"dataset.current={self.profile_name}",
                f"dataset.num_cells={num_cells}",
                # Architecture (varies across trials)
                f"model.batch_size={self.batch_size}",
                f"model.emsize={self.emsize}",
                f"model.d_hid={self.d_hid}",
                f"model.nhead={self.nhead}",
                f"model.nlayers={self.nlayers}",
                f"model.output_dim={self.output_dim}",
                f"model.dropout={self.dropout}",
                # Optimizer (held fixed across trials)
                f"optimizer.max_lr={self.max_lr}",
                f"optimizer.weight_decay={hp.get('weight_decay', 0.01)}",
                # Experiment bookkeeping
                f"experiment.name=state_{self.profile_name}_{self.model_name}",
                f"experiment.port={self._get_unique_port()}",
                f"experiment.checkpoint.path={self.checkpoint_dir}",
                # Hard cap on total gradient steps (via profiler path).
                "experiment.profile.enable_profiler=true",
                f"experiment.profile.max_steps={OPTIMIZER_STEPS}",
                f"experiment.profile.profile_steps=[0,{OPTIMIZER_STEPS}]",
                # Logging cadence — matches scaling_laws/algo/state.py.
                f"experiment.val_check_interval={VAL_CHECK_INTERVAL}",
                "experiment.limit_val_batches=50",
                "+experiment.log_every_n_steps=10",
                f"experiment.checkpoint.every_n_train_steps={VAL_CHECK_INTERVAL}",
                "experiment.checkpoint.save_top_k=-1",
                "experiment.checkpoint.monitor=validation/val_loss",
                "+experiment.checkpoint.save_last=false",
                # Defaults we must flip for the sweep
                "experiment.early_stopping.enable=false",
                "wandb.enable=false",
                "validations.diff_exp.enable=false",
                "validations.perturbation.enable=false",
            ]
            try:
                subprocess.run(cmd_fit, cwd=str(self.state_package_dir), env=env, check=True)
            finally:
                self._save_loss_curves()
                self._save_final_losses()

    trial_dir = OUTPUT_DIR / subpath
    trial_dir.mkdir(parents=True, exist_ok=True)
    arch_keys = ("emsize", "d_hid", "nhead", "nlayers", "output_dim", "pad_length")
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
        "arch": {k: int(trial[k]) for k in arch_keys},
        "fixed_hparams": FIXED_HPARAMS,
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
        extra_hparams = {"weight_decay": FIXED_HPARAMS["weight_decay"]}
        arch_kwargs = {k: trial[k] for k in arch_keys}
        print(f"[{trial_name}] phase=construct  base_dir={base_dir}", flush=True)
        model = TunableState(
            base_dir=str(base_dir),
            device=device,
            dataset_name=DATASET,
            seed=SEED,
            model_name=trial_name,
            subpath=subpath,
            max_lr=FIXED_HPARAMS["max_lr"],
            dropout=FIXED_HPARAMS["dropout"],
            batch_size=FIXED_HPARAMS["batch_size"],
            extra_hparams=extra_hparams,
            **arch_kwargs,
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
        result_doc = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in result.items()}
        (trial_dir / "result.json").write_text(json.dumps(result_doc, indent=2, default=str))
        print(f"[{trial_name}] result -> {trial_dir / 'result.json'}  status={result['status']}", flush=True)

    return result


# ── Orchestrator ────────────────────────────────────────────────────────

def detect_gpus() -> list[int]:
    """Detect visible GPU indices via nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def main():
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

    model_configs = generate_trials()
    print(f"Generated {len(model_configs)} model-size configs:")
    for t in model_configs:
        approx_params = t["nlayers"] * (4 * t["emsize"] ** 2 + 2 * t["emsize"] * t["d_hid"])
        print(f"  {TRIAL_PREFIX}_{t['trial_id']:02d}: "
              f"emsize={t['emsize']} d_hid={t['d_hid']} nhead={t['nhead']} "
              f"nlayers={t['nlayers']} output_dim={t['output_dim']}  "
              f"~{approx_params/1e6:.1f}M xfmr-body params")

    jobs = []
    for cfg in model_configs:
        for size in SIZES:
            for quality in QUALITIES:
                job = {**cfg}
                job["size"] = size
                job["quality"] = quality
                tid_tag = f"{TRIAL_PREFIX}_{cfg['trial_id']:02d}"
                sz_tag = str(size)
                q_tag = str(quality)
                job["trial_name"] = f"{tid_tag}_{sz_tag}_{q_tag}"
                job["subpath"] = f"{tid_tag}/{sz_tag}/{q_tag}"
                jobs.append(job)
    random.Random(SEED).shuffle(jobs)

    total_jobs = len(jobs)
    print(f"\nTotal jobs: {len(model_configs)} configs x {len(SIZES)} sizes x "
          f"{len(QUALITIES)} qualities = {total_jobs}  "
          f"(optimizer_steps={OPTIMIZER_STEPS} per trial)")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    free_slots = list(slots)
    results: list[dict] = []
    pending = list(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}

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

    out_csv = output_dir / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults -> {out_csv}")

    arch_keys = ("emsize", "d_hid", "nhead", "nlayers", "output_dim", "pad_length")
    for trial_id in sorted(df["trial_id"].unique()):
        sub = df[df["trial_id"] == trial_id]
        row0 = sub.iloc[0]
        config = {"trial_id": int(trial_id),
                  "arch": {k: int(row0[k]) for k in arch_keys if k in row0.index},
                  "fixed_hparams": FIXED_HPARAMS}
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

    ok_df = df[df["status"] == "ok"]
    mi_cols = [c for c in ok_df.columns if c.startswith("mi_")]
    if not mi_cols:
        print("\nNo MI columns to summarize.")
        return
    agg_cols = {c: "mean" for c in mi_cols}
    primary_mi = mi_cols[0]
    summary = ok_df.groupby("trial_id").agg(agg_cols).sort_values(primary_mi, ascending=False)

    print(f"\nMean MI across qualities per model-size config (sorted by {primary_mi}):")
    for trial_id, row in summary.iterrows():
        row0 = df[df["trial_id"] == trial_id].iloc[0]
        mi_str = " ".join(f"{c}={row[c]:.3f}" for c in mi_cols if pd.notna(row[c]))
        print(f"  trial_{int(trial_id):02d}: {mi_str}  "
              f"(emsize={int(row0['emsize'])} nlayers={int(row0['nlayers'])})")


if __name__ == "__main__":
    main()
