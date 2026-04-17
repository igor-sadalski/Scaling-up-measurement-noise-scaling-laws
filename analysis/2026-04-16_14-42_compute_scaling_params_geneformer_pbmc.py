"""Parameter- and quality-scaling analysis for Geneformer on PBMC (size=100000).

For each of 5 architectures log-spaced from ~1.5M up to ~100M parameters, trains
the model at each of 10 log-spaced quality levels (1.0 down to ~0.00123) and
records the test MLM loss. Produces a long-form CSV with one row per
(architecture, quality) cell. A companion notebook
(``2026-04-16_14-42_plotting_scaling_params_geneformer_pbmc.ipynb``) reads the CSV
and plots loss vs. parameter count coloured by quality.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate modeling && \
    python analysis/2026-04-16_14-42_compute_scaling_params_geneformer_pbmc.py --device 0

Optional flags:
    --configs IDX [IDX ...]    train only a subset of configs (0-indexed)
    --qualities Q [Q ...]      train only a subset of quality values
    --max-epochs N             override per-config epoch count (default 3)
    --resume                   skip (config, quality) pairs with a saved test loss
"""
   
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_from_disk
from geneformer import GeneformerPretrainer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    EarlyStoppingCallback,
    TrainingArguments,
)

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


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


_log_f = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
print(f"Logging to {LOG_PATH}")


# ----------------------------------------------------------------------------
# Paths (PBMC @ 100k cells, sweeping 10 qualities)
# ----------------------------------------------------------------------------
DATA_ROOT = Path("/home/igor/noise_scaling/data")
DATASET = "PBMC"
SIZE = 100_000

TOKEN_DICT_PATH = DATA_ROOT / DATASET / "utils" / "token_dict.pkl"
SUMMARY_CSV = (
    Path(__file__).parent
    / "2026-04-16_14-42_scaling_params_geneformer_pbmc_results.csv"
)

# Ten log-spaced qualities, matching run_pbmc_whole.py.
QUALITIES: list[float] = [
    round(q, 7) for q in np.logspace(0, np.log10(10 / 8100), 10)
]


def paths_for_quality(quality: float) -> dict[str, Path]:
    """Compute all per-(size, quality) dataset and output paths."""
    base_dir = DATA_ROOT / DATASET / str(SIZE) / str(quality)
    return {
        "base_dir": base_dir,
        "train_tok": base_dir / "preprocessed" / "tokenized.dataset",
        "val_tok": DATA_ROOT / DATASET / "validation" / str(quality) / "preprocessed" / "tokenized.dataset",
        "test_tok": DATA_ROOT / DATASET / "test" / str(quality) / "preprocessed" / "tokenized.dataset",
        "lengths": base_dir / "preprocessed" / "lengths.pkl",
        "results_root": base_dir / "results" / "Geneformer_scaling",
    }

SEED = 42
MAX_INPUT_SIZE = 512


@dataclass
class ArchConfig:
    """A single Geneformer architecture point on the parameter-scaling curve."""

    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int


# 5 architectures roughly log-spaced from ~1.5M to ~100M total params.
# Approximate total-param targets (vocab≈20.7k, max_pos=512):
#   tiny ≈ 1.5M, small ≈ 4M, medium ≈ 12M, large ≈ 35M, xlarge ≈ 100M (BERT-base).
CONFIGS: list[ArchConfig] = [
    ArchConfig("tiny", hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128),
    ArchConfig("small", hidden_size=160, num_hidden_layers=3, num_attention_heads=4, intermediate_size=640),
    ArchConfig("medium", hidden_size=384, num_hidden_layers=3, num_attention_heads=6, intermediate_size=1024),
    ArchConfig("large", hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048),
    ArchConfig("xlarge", hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ArchConfig, vocab_size: int, pad_id: int) -> BertForMaskedLM:
    bert_cfg = BertConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_act="relu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.02,
        hidden_dropout_prob=0.02,
        max_position_embeddings=MAX_INPUT_SIZE,
        model_type="bert",
        pad_token_id=pad_id,
        vocab_size=vocab_size,
    )
    return BertForMaskedLM(bert_cfg)


def train_one_config(
    cfg: ArchConfig,
    quality: float,
    device: int,
    max_epochs: int,
    token_dictionary: dict,
    per_device_train_bs: int = 64,
    per_device_eval_bs: int = 100,
    warmup_steps: int = 2_000,
    max_lr: float = 1e-3,
    weight_decay: float = 0.001,
    save_steps: int = 1000,
    early_stopping_patience: int = 5,
) -> dict:
    """Train one (architecture, quality) point and return a metrics dict."""
    paths = paths_for_quality(quality)
    run_dir = paths["results_root"] / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model"
    test_loss_path = model_path / "test_loss.txt"

    model = build_model(cfg, vocab_size=len(token_dictionary), pad_id=token_dictionary["<pad>"])
    n_params = count_parameters(model)
    n_non_embed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and "embeddings" not in n
    )
    print(f"\n=== {cfg.name} q={quality} | H={cfg.hidden_size} L={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} ffn={cfg.intermediate_size} ===")
    print(f"  total params: {n_params:,}   non-embedding: {n_non_embed:,}")

    wandb.init(
        project="geneformer-param-scaling",
        name=f"PBMC_{SIZE}_q{quality}_{cfg.name}",
        config={**asdict(cfg), "total_params": n_params, "non_embedding_params": n_non_embed,
                "dataset": DATASET, "size": SIZE, "quality": quality, "max_epochs": max_epochs},
        reinit=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.train()

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=max_lr,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        num_train_epochs=max_epochs,
        save_strategy="steps",
        logging_steps=save_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="wandb",
        eval_steps=save_steps,
        save_steps=save_steps,
        logging_dir=str(run_dir),
        seed=SEED,
    )

    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        train_dataset=load_from_disk(str(paths["train_tok"])),
        eval_dataset=load_from_disk(str(paths["val_tok"])),
        example_lengths_file=str(paths["lengths"]),
        token_dictionary=token_dictionary,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    t0 = time.time()
    train_output = trainer.train()
    train_time = time.time() - t0
    trainer.save_model(str(model_path))

    # Pull final train / eval loss from the log history
    hist = pd.DataFrame(trainer.state.log_history)
    final_train_loss = float(hist["loss"].dropna().iloc[-1]) if "loss" in hist else float("nan")
    final_eval_loss = float(hist["eval_loss"].dropna().iloc[-1]) if "eval_loss" in hist else float("nan")
    best_eval_loss = float(hist["eval_loss"].dropna().min()) if "eval_loss" in hist else float("nan")

    # Compute test loss
    test_args = TrainingArguments(
        output_dir=str(run_dir / "eval_tmp"),
        per_device_eval_batch_size=per_device_eval_bs,
        do_eval=True,
        report_to="none",
        seed=SEED,
    )
    test_trainer = GeneformerPretrainer(
        model=trainer.model,
        args=test_args,
        eval_dataset=load_from_disk(str(paths["test_tok"])),
        token_dictionary=token_dictionary,
        example_lengths_file=str(paths["lengths"]),
    )
    test_metrics = test_trainer.evaluate()
    test_loss = float(test_metrics["eval_loss"])
    test_loss_path.parent.mkdir(parents=True, exist_ok=True)
    test_loss_path.write_text(f"{test_loss:.6f}")

    wandb.log({
        "test_loss": test_loss,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "total_parameters": n_params,
        "non_embedding_parameters": n_non_embed,
        "train_time_s": train_time,
    })
    wandb.finish()

    # Free GPU memory before next config
    del trainer, test_trainer, model
    torch.cuda.empty_cache()

    return {
        "name": cfg.name,
        "quality": quality,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "total_params": n_params,
        "non_embedding_params": n_non_embed,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "test_loss": test_loss,
        "train_time_s": train_time,
        "train_runtime": train_output.metrics.get("train_runtime"),
        "train_samples_per_second": train_output.metrics.get("train_samples_per_second"),
        "model_dir": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--configs", type=int, nargs="+", default=None,
                        help="Indexes into CONFIGS list to run (default: all)")
    parser.add_argument("--qualities", type=float, nargs="+", default=None,
                        help="Quality values to run (default: all 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (config, quality) pairs with existing test_loss.txt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    with open(TOKEN_DICT_PATH, "rb") as f:
        token_dictionary = pickle.load(f)

    indices = args.configs if args.configs is not None else list(range(len(CONFIGS)))
    selected_cfgs = [CONFIGS[i] for i in indices]
    selected_qualities = args.qualities if args.qualities is not None else QUALITIES

    # Load existing results so a resumed run keeps prior rows
    if SUMMARY_CSV.exists():
        results = pd.read_csv(SUMMARY_CSV).to_dict("records")
    else:
        results = []

    # Sweep in (config, quality) order so each architecture produces a full
    # noise curve before moving to the next one — nicer for intermediate plots.
    for cfg in selected_cfgs:
        for quality in selected_qualities:
            test_loss_file = (
                paths_for_quality(quality)["results_root"] / cfg.name / "model" / "test_loss.txt"
            )
            if args.resume and test_loss_file.exists():
                print(f"[resume] skipping {cfg.name} q={quality}; {test_loss_file} already exists")
                continue

            row = train_one_config(
                cfg, quality, args.device, args.max_epochs, token_dictionary
            )

            # Drop any prior row for the same (config, quality), then append fresh
            results = [
                r for r in results
                if not (r.get("name") == cfg.name and float(r.get("quality", -1)) == float(quality))
            ]
            results.append(row)
            df = (
                pd.DataFrame(results)
                .sort_values(["total_params", "quality"])
                .reset_index(drop=True)
            )
            df.to_csv(SUMMARY_CSV, index=False)
            print(f"  -> wrote {SUMMARY_CSV} ({len(df)} rows)")

    print("\nDone. Summary:")
    print(pd.read_csv(SUMMARY_CSV).to_string(index=False))


if __name__ == "__main__":
    main()
