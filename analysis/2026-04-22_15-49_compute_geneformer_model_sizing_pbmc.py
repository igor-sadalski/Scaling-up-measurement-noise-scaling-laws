"""Geneformer model-size sweep on PBMC (model_size x quality grid).

Goal: study how Geneformer's training behavior scales with parameter count,
holding data fixed at the largest available PBMC size and varying only
architecture. We define 7 model configs centered on a CONTROL whose
architecture matches the scaling_laws.algo.geneformer.Geneformer defaults
used by the production all-datasets run: three configs smaller than the
control (the two smallest are intentionally tiny to probe where the
scaling law breaks), the control itself, and three configs larger than it.

Mirrors the STATE sweep (analysis/2026-04-20_14-31_compute_state_model_sizing_
pbmc.py). For each config we run all qualities on the largest dataset size.

Architecture ratios are kept constant across configs:
    intermed_size    = 2 * num_embed_dim    (FFN hidden = 2 x hidden)
    num_attn_heads   = num_embed_dim / 64   (head dim fixed at 64)
    max_input_size   = 512                  (constant)

Only `num_embed_dim` and `num_layers` vary; the CONTROL is
`num_embed_dim=256, num_layers=3` to match Geneformer()'s defaults.

Hyperparameters held fixed at Geneformer defaults across all configs (see
scaling_laws/algo/geneformer.py): max_lr=1e-3, per_device_train_bs=64,
weight_decay=0.001, warmup_steps=5000, activation=relu, dropouts=0.02.

Smoke-test mode: if RUN_CONTROL_ONLY is True, only the CONTROL trial runs.
Flip to False to run the full size sweep.

Training protocol:
    - step budget: OPTIMIZER_STEPS=15000, enforced via TrainingArguments.max_steps
      (overrides num_train_epochs). Same budget as the STATE model-size sweep
      so the two figures are directly comparable.
    - early stopping ALWAYS ON, metric=eval_loss, patience=5, eval every 1000 steps
    - eval/save every 1000 steps (save_total_limit kept high so ES can recover any checkpoint)
    - load_best_model_at_end=True so trainer.save_model() writes the best checkpoint

Parallelism: JOBS_PER_GPU=1 across all visible GPUs (largest config is ~38M
trainable params -- fits on 1 H100 comfortably but conservative scheduling
keeps the sweep simple).

Outputs:
    $NOISE_SCALING_OUTPUT_BASE/model_sizing_geneformer/  (default ~/noise_scaling/data/other/model_sizing_geneformer)
        sweep_results.csv             # full results table
        model_sizing_geneformer_NN.yaml  # per-config YAML with arch + per-quality results
        model_sizing_geneformer_NN/<size>/<quality>/
            config.json
            result.json
            trainer_state.json        # HF Trainer log_history (train/eval loss curves)
            checkpoint-*/             # HF checkpoints (save_total_limit=500)
            model.safetensors         # best checkpoint after load_best_model_at_end
            embeddings.csv
            MI/<seed>/Y_<signal>_<quality>_geneformer/lmi_mutual_information.txt
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

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
SIGNAL_COLUMNS = ["celltype.l3", "protein_counts"]  # matches run_pbmc_whole.py

from scaling_laws.paths import DATA_DIR, OUTPUT_BASE
OUTPUT_DIR = OUTPUT_BASE / "model_sizing_geneformer"

TRIAL_PREFIX = "model_sizing_geneformer"
JOBS_PER_GPU = 2
SEED = 42
# Hard cap on total optimizer (gradient) steps per trial -- same budget as the
# STATE model-size sweep so loss curves are directly comparable across algos.
OPTIMIZER_STEPS = 40_000
EVAL_SAVE_STEPS = 1000
LOG_EVERY_N_STEPS = 10


# -- Hyperparameters held fixed across all model sizes (Geneformer defaults) --
# Match scaling_laws/algo/geneformer.py Geneformer.train() so the CONTROL trial
# is directly comparable to the production all-datasets run.
FIXED_HPARAMS = {
    "max_lr":                        1e-3,
    "per_device_train_bs":           64,   # 32 * 2 in prod
    "per_device_eval_bs":            100,
    "weight_decay":                  0.001,
    "warmup_steps":                  5000,
    "activ_fn":                      "relu",
    "initializer_range":             0.02,
    "layer_norm_eps":                1e-12,
    "attention_probs_dropout_prob":  0.02,
    "hidden_dropout_prob":           0.02,
    "lr_schedule_fn":                "linear",
    "early_stopping_patience":       5,
}


# -- Model-size configurations ------------------------------------------
# 7 configs centered on the CONTROL (num_embed_dim=256, num_layers=3 --
# Geneformer defaults): three smaller (two of them tiny, to probe where the
# scaling law breaks), the control, and three larger.
# Architecture ratios held constant for the main sweep: intermed_size =
# 2*num_embed_dim, num_attn_heads = num_embed_dim / 64 (head dim = 64),
# max_input_size = 512. The two tiniest configs (trial 0, trial 1) break the
# head_dim=64 invariant on purpose so the model can get small enough to
# expose a regime change.
#
# Approximate BERT body param counts (excluding embeddings layer):
# per-layer body ~ 12 * hidden^2 (standard BertLayer with intermed=2*hidden),
# so body params ~ 12 * num_embed_dim^2 * num_layers.
#   trial 0 -- num_embed_dim= 32, num_layers=1  ~  0.012M body params (tiny -- expected to break scaling)
#   trial 1 -- num_embed_dim= 64, num_layers=2  ~  0.098M body params (very small)
#   trial 2 -- num_embed_dim=128, num_layers=2  ~  0.39M  body params (smaller)
#   trial 3 -- num_embed_dim=256, num_layers=3  ~  2.36M  body params (CONTROL -- Geneformer defaults)
#   trial 4 -- num_embed_dim=384, num_layers=4  ~  7.08M  body params (larger)
#   trial 5 -- num_embed_dim=512, num_layers=6  ~ 18.87M  body params (larger)
#   trial 6 -- num_embed_dim=768, num_layers=8  ~ 56.62M  body params (larger)
MODEL_CONFIGS = [
    {"num_embed_dim":  32, "intermed_size":   64, "num_attn_heads":  1, "num_layers": 1, "max_input_size": 512},
    {"num_embed_dim":  64, "intermed_size":  128, "num_attn_heads":  1, "num_layers": 2, "max_input_size": 512},
    # {"num_embed_dim": 128, "intermed_size":  256, "num_attn_heads":  2, "num_layers": 2, "max_input_size": 512},
    # {"num_embed_dim": 256, "intermed_size":  512, "num_attn_heads":  4, "num_layers": 3, "max_input_size": 512},
    # {"num_embed_dim": 384, "intermed_size":  768, "num_attn_heads":  6, "num_layers": 4, "max_input_size": 512},
    # {"num_embed_dim": 512, "intermed_size": 1024, "num_attn_heads":  8, "num_layers": 6, "max_input_size": 512},
    # {"num_embed_dim": 768, "intermed_size": 1536, "num_attn_heads": 12, "num_layers": 8, "max_input_size": 512},
]
CONTROL_TRIAL_ID = 3  # index into MODEL_CONFIGS for the Geneformer-defaults config

# Smoke-test mode: when True, only the CONTROL trial runs.
RUN_CONTROL_ONLY = False


def generate_trials() -> list[dict]:
    """Wrap MODEL_CONFIGS into trial dicts with sequential trial_ids.

    If RUN_CONTROL_ONLY is True, returns only the CONTROL trial.
    """
    all_trials = [{"trial_id": i, **cfg} for i, cfg in enumerate(MODEL_CONFIGS)]
    if RUN_CONTROL_ONLY:
        return [t for t in all_trials if t["trial_id"] == CONTROL_TRIAL_ID]
    return all_trials


# -- Per-trial runner (executed in a subprocess) -------------------------

def run_trial(trial: dict, device: int) -> dict:
    """Train Geneformer with `trial`'s architecture on `device`."""
    # `scaling_laws` is pip-installed (editable) in the launching env;
    # ProcessPoolExecutor workers inherit sys.path via fork.
    from scaling_laws.algo.geneformer import Geneformer

    trial_name = trial["trial_name"]
    subpath = trial["subpath"]
    size = trial["size"]
    quality = trial["quality"]
    base_dir = DATA_DIR / DATASET / str(size) / str(quality)

    class TunableGeneformer(Geneformer):
        """Geneformer variant that exposes all arch params as constructor
        overrides. Each trial writes to its own OUTPUT_DIR/<subpath>/."""

        def __init__(self, model_name: str = "model", subpath: str | None = None,
                     num_embed_dim: int = 256, intermed_size: int = 512,
                     num_attn_heads: int = 4, num_layers: int = 3,
                     max_input_size: int = 512, **kw):
            super().__init__(model_name=model_name, **kw)
            # Keep method_name == "Geneformer" so BaseAlgorithm.mutual_information
            # still picks up signal files with the _geneformer suffix
            # (its filter is hard-coded to that string).
            self.method_name = "Geneformer"
            # Re-route every artifact to OUTPUT_DIR/<subpath>/. Matches the
            # STATE sweep's TunableState layout: model_path is the trial dir
            # and save_folder_path is its parent so MI's signal_folder
            # (save_folder_path / model_path.name / MI / ...) lands back in
            # model_path/MI/... exactly as desired.
            self.model_name = model_name
            self.subpath = subpath if subpath is not None else model_name
            self.model_path = OUTPUT_DIR / self.subpath
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.save_folder_path = self.model_path.parent
            self.save_folder_path.mkdir(parents=True, exist_ok=True)
            self.embeddings_path = self.model_path / "embeddings.csv"
            self.test_loss_path = self.model_path / "test_loss.txt"
            # Swept architecture knobs
            self.num_embed_dim = int(num_embed_dim)
            self.intermed_size = int(intermed_size)
            self.num_attn_heads = int(num_attn_heads)
            self.num_layers = int(num_layers)
            self.max_input_size = int(max_input_size)
            # The EmbExtractor in embed() reads this to pick the first `embed_dim`
            # columns of the tsv as the embedding block; keep it in sync with hidden.
            self.embed_dim = int(num_embed_dim)

        def train(self) -> None:
            """Train BertForMaskedLM with the swept arch and a hard step budget.

            Mirrors Geneformer.train() but:
              - exposes num_embed_dim / num_layers / num_attn_heads / intermed_size
                / max_input_size as instance attributes that came from the sweep,
              - uses max_steps=OPTIMIZER_STEPS instead of num_train_epochs,
              - sends checkpoints / trainer_state.json to self.model_path (not
                to the shared save_folder_path), so trials don't clobber each other,
              - disables wandb reporting.
            """
            import torch
            from datasets import load_from_disk
            from transformers import (
                BertConfig,
                BertForMaskedLM,
                EarlyStoppingCallback,
                TrainingArguments,
            )
            from geneformer import GeneformerPretrainer

            model_type = "bert"
            hp = FIXED_HPARAMS

            with open(self.token_dictionary_path, "rb") as fp:
                token_dictionary = pickle.load(fp)

            bert_config = BertConfig(
                hidden_size=self.num_embed_dim,
                num_hidden_layers=self.num_layers,
                initializer_range=hp["initializer_range"],
                layer_norm_eps=hp["layer_norm_eps"],
                attention_probs_dropout_prob=hp["attention_probs_dropout_prob"],
                hidden_dropout_prob=hp["hidden_dropout_prob"],
                intermediate_size=self.intermed_size,
                hidden_act=hp["activ_fn"],
                max_position_embeddings=self.max_input_size,
                model_type=model_type,
                num_attention_heads=self.num_attn_heads,
                pad_token_id=token_dictionary.get("<pad>"),
                vocab_size=len(token_dictionary),
            )
            self.model = BertForMaskedLM(bert_config)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n[{self.model_name}] Model parameters: {num_params:,} "
                  f"(hidden={self.num_embed_dim}, layers={self.num_layers}, "
                  f"heads={self.num_attn_heads}, intermed={self.intermed_size})")

            training_args = TrainingArguments(
                learning_rate=hp["max_lr"],
                do_train=True,
                do_eval=True,
                evaluation_strategy="steps",
                group_by_length=True,
                length_column_name="length",
                disable_tqdm=False,
                lr_scheduler_type=hp["lr_schedule_fn"],
                warmup_steps=hp["warmup_steps"],
                weight_decay=hp["weight_decay"],
                per_device_train_batch_size=hp["per_device_train_bs"],
                per_device_eval_batch_size=hp["per_device_eval_bs"],
                max_steps=OPTIMIZER_STEPS,
                save_strategy="steps",
                logging_strategy="steps",
                logging_steps=LOG_EVERY_N_STEPS,
                output_dir=str(self.model_path),
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_dir=str(self.model_path),
                load_best_model_at_end=True,
                save_total_limit=500,
                report_to="none",
                eval_steps=EVAL_SAVE_STEPS,
                save_steps=EVAL_SAVE_STEPS,
                seed=SEED,
            )

            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=hp["early_stopping_patience"],
            )

            if torch.cuda.is_available():
                self.model = self.model.to("cuda:0").train()
                print(f"[{self.model_name}] moved to GPU "
                      f"(device {self.device} visible as cuda:0)")

            self.trainer = GeneformerPretrainer(
                model=self.model,
                args=training_args,
                train_dataset=load_from_disk(self.geneformer_train_path),
                eval_dataset=load_from_disk(self.geneformer_validation_path),
                example_lengths_file=str(self.lengths_path),
                token_dictionary=token_dictionary,
                callbacks=[early_stopping_callback],
            )

            self.trainer.train()
            # load_best_model_at_end=True means the model is already the best
            # checkpoint by val loss; save it as the canonical model.
            self.trainer.save_model(self.model_path)
            # Persist the final trainer_state so the plotting notebook can read
            # log_history (train + eval loss curves) from a stable location.
            self.trainer.state.save_to_json(self.model_path / "trainer_state.json")

    trial_dir = OUTPUT_DIR / subpath
    trial_dir.mkdir(parents=True, exist_ok=True)
    arch_keys = ("num_embed_dim", "intermed_size", "num_attn_heads", "num_layers", "max_input_size")
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
        arch_kwargs = {k: trial[k] for k in arch_keys}
        print(f"[{trial_name}] phase=construct  base_dir={base_dir}", flush=True)
        model = TunableGeneformer(
            base_dir=str(base_dir),
            lengths_path=str(base_dir / "preprocessed" / "lengths.pkl"),
            signal_columns=SIGNAL_COLUMNS,
            device=device,
            dataset_name=DATASET,
            seed=SEED,
            model_name=trial_name,
            subpath=subpath,
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


# -- Orchestrator --------------------------------------------------------

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
            tokenized = base_dir / "preprocessed" / "tokenized.dataset"
            assert tokenized.exists(), (
                f"Geneformer tokenized dataset missing: {tokenized}. "
                f"Run prepare_data(...) for {DATASET} first."
            )
            lengths = base_dir / "preprocessed" / "lengths.pkl"
            assert lengths.exists(), f"Geneformer lengths.pkl missing: {lengths}"
    token_dict = DATA_DIR / DATASET / "utils" / "token_dict.pkl"
    assert token_dict.exists(), f"Token dictionary missing: {token_dict}"
    print(f"All {len(SIZES)} sizes x {len(QUALITIES)} qualities verified.")

    gpus = detect_gpus()
    slots = gpus * JOBS_PER_GPU
    max_workers = len(slots)
    print(f"GPUs={gpus}  jobs_per_gpu={JOBS_PER_GPU}  slots={max_workers}")

    model_configs = generate_trials()
    print(f"Generated {len(model_configs)} model-size configs:")
    for t in model_configs:
        # Rough transformer-body param count (12 * hidden^2 * layers for a
        # BERT-style block with FFN = 2 * hidden; understates total by the
        # embedding + MLM-head tables).
        approx_params = 12 * t["num_embed_dim"] ** 2 * t["num_layers"]
        print(f"  {TRIAL_PREFIX}_{t['trial_id']:02d}: "
              f"hidden={t['num_embed_dim']} layers={t['num_layers']} "
              f"heads={t['num_attn_heads']} intermed={t['intermed_size']}  "
              f"~{approx_params/1e6:.1f}M body params (excl. embeddings)")

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

    arch_keys = ("num_embed_dim", "intermed_size", "num_attn_heads", "num_layers", "max_input_size")
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
              f"(hidden={int(row0['num_embed_dim'])} layers={int(row0['num_layers'])})")


if __name__ == "__main__":
    main()
