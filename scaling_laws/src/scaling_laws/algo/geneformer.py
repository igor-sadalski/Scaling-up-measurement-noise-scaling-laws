import os
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from latentmi import lmi
from geneformer import GeneformerPretrainer, EmbExtractor
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
import pickle
import wandb
from .abc import BaseAlgorithm
import anndata as ad


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Geneformer(BaseAlgorithm):
    def __init__(
        self,
        base_dir,
        lengths_path: str | None = None,
        signal_columns: list[str] | None = None,
        device: int = 0,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 100,
        dataset_name: str | None = None,
        model_name: str = "model",
        seed: int = 42,
    ):
        super().__init__(base_dir, device, model_name=model_name, seed=seed)
        self.model: BertForMaskedLM | None = None
        self.trainer: GeneformerPretrainer | None = None
        self.token_dictionary_path: Path = Path(self.base_dir.parent.parent / "utils" / "token_dict.pkl")
        self.lengths_path: Path | None = Path(lengths_path) if lengths_path is not None else None
        self.signal_columns: list[str] = signal_columns if signal_columns is not None else ["celltype.l3"]
        self.max_epochs: int = max_epochs
        self.early_stopping_patience: int = early_stopping_patience
        self.early_stopping_threshold: float = early_stopping_threshold
        self.dataset_name: str = dataset_name or "unknown"

        self.per_device_eval_bs: int = 100  # idealy this should be dynamic adjusted
        self.per_device_train_bs: int = 32 * 2  # per device batch size, thats pushing it already...
        self.embed_dim: int = 256

        self.geneformer_train_path = self.train_data_path / "tokenized.dataset"
        self.geneformer_test_path = self.test_data_path / "tokenized.dataset"
        self.geneformer_validation_path = self.validation_data_path / "tokenized.dataset"

    def train(
        self,
    ) -> None:
        model_type = "bert"
        max_input_size = 512
        num_layers = 3
        num_attn_heads = 4
        num_embed_dim = self.embed_dim
        intermed_size = num_embed_dim * 2
        activ_fn = "relu"
        initializer_range = 0.02
        layer_norm_eps = 1e-12
        attention_probs_dropout_prob = 0.02
        hidden_dropout_prob = 0.02
        per_device_train_bs = self.per_device_train_bs  # <- 64
        max_lr = 1e-3
        lr_schedule_fn = "linear"
        warmup_steps = 5_000
        epochs = self.max_epochs
        optimizer = "adamw"
        weight_decay = 0.001
        per_device_eval_bs = self.per_device_eval_bs  # <- 100

        wandb.init(
            project="geneformer-scaling-laws",
            name=f"{self.dataset_name}_size{self.size}_quality{self.quality}",
            config={
                "dataset": self.dataset_name,
                "size": self.size,
                "quality": self.quality,
                "model_type": model_type,
                "max_input_size": max_input_size,
                "num_layers": num_layers,
                "num_attn_heads": num_attn_heads,
                "num_embed_dim": num_embed_dim,
                "intermed_size": intermed_size,
                "activ_fn": activ_fn,
                "initializer_range": initializer_range,
                "layer_norm_eps": layer_norm_eps,
                "attention_probs_dropout_prob": attention_probs_dropout_prob,
                "hidden_dropout_prob": hidden_dropout_prob,
                "per_device_train_bs": per_device_train_bs,
                "max_lr": max_lr,
                "lr_schedule_fn": lr_schedule_fn,
                "warmup_steps": warmup_steps,
                "epochs": epochs,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "per_device_eval_bs": per_device_eval_bs,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": 15,
                "signal_columns": self.signal_columns,
            },
        )

        with open(self.token_dictionary_path, "rb") as fp:
            token_dictionary = pickle.load(fp)

        config = {
            "hidden_size": num_embed_dim,
            "num_hidden_layers": num_layers,
            "initializer_range": initializer_range,
            "layer_norm_eps": layer_norm_eps,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "intermediate_size": intermed_size,
            "hidden_act": activ_fn,
            "max_position_embeddings": max_input_size,
            "model_type": model_type,
            "num_attention_heads": num_attn_heads,
            "pad_token_id": token_dictionary.get("<pad>"),
            "vocab_size": len(token_dictionary),
        }

        config = BertConfig(**config)
        self.model = BertForMaskedLM(config)

        # Print number of parameters
        num_params = count_parameters(self.model)
        print(f"\nModel Parameters Summary:")
        print(f"Total trainable parameters: {num_params:,}")
        print(f"Number of layers: {num_layers}")
        print(f"Hidden dimension: {num_embed_dim}")
        print(f"Attention heads: {num_attn_heads}\n")

        # Log parameters to wandb
        wandb.log({"total_parameters": num_params, "parameters_millions": num_params / 1_000_000})

        save_steps = 1000

        training_args = {
            "learning_rate": max_lr,
            "do_train": True,
            "do_eval": True,
            "evaluation_strategy": "steps",
            "group_by_length": True,
            "length_column_name": "length",
            "disable_tqdm": False,
            "lr_scheduler_type": lr_schedule_fn,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "per_device_train_batch_size": per_device_train_bs,
            "per_device_eval_batch_size": per_device_eval_bs,
            "num_train_epochs": epochs,
            "save_strategy": "steps",
            "logging_steps": save_steps,
            "output_dir": self.save_folder_path,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "logging_dir": self.save_folder_path,
            "load_best_model_at_end": True,
            "save_total_limit": 500,
            "report_to": "wandb",
            "eval_steps": save_steps,
            "save_steps": save_steps,
        }
        training_args = TrainingArguments(**training_args)

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
        )

        if torch.cuda.is_available() and self.model is not None:
            self.model = self.model.to("cuda:0")

        if self.model is not None:
            self.model = self.model.train()

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
        self.trainer.save_model(self.model_path)
        wandb.finish()

    def embed(self, inference_batch_size: int | None = 50) -> np.ndarray:

        self.geneformer_test_path = self.test_data_path / "tokenized.dataset"

        if self.dataset_name.lower() == "larry":
            emb_label = ["index", "clone", "time"]
        else:
            emb_label = []
            available_signals = {}
            for col in self.signal_columns:
                if col == "protein_counts":
                    adata = ad.read_h5ad(self.train_data_path / "preprocessed.h5ad", backed="r")
                    new_label = [c for c in adata.obs.columns if c.startswith("prot_")]
                    emb_label.extend(new_label)
                else:
                    new_label = [col]
                    emb_label.extend(new_label)

                available_signals[col] = len(new_label)

        embex = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",
            cell_emb_style="mean_pool",
            gene_emb_style="mean_pool",
            emb_layer=-1,
            forward_batch_size=inference_batch_size if inference_batch_size is not None else self.per_device_eval_bs,
            nproc=30,  # how to speed up data prefetching here
            token_dictionary_file=str(self.token_dictionary_path),
            max_ncells=None,  #!!!!
            emb_label=emb_label,
        )

        embs = embex.extract_embs(
            model_directory=str(self.model_path),
            input_data_file=str(self.geneformer_test_path),
            output_directory=str(self.save_folder_path),
            output_prefix="embeddings",
            output_torch_embs=True,
        )

        X = embs[0].values[:, : self.embed_dim]
        Ys = embs[0].values[:, self.embed_dim :]

        pd.DataFrame(X).to_csv(self.embeddings_path, index=False)

        index = 0
        if self.dataset_name.lower() == "merfish":
            signal_path = (
                self.test_signal_path.parent / f"{self.quality}" / "signals" / f"Y_ng_idx_{self.quality}_geneformer.csv"
            )
            pd.DataFrame(Ys).to_csv(signal_path, index=False)
        elif self.dataset_name.lower() == "larry":

            signal_path = (
                self.test_signal_path.parent / f"{self.quality}" / "signals" / f"Y_clone_{self.quality}_geneformer.csv"
            )
            pd.DataFrame(Ys).to_csv(signal_path, index=False)
        else:
            for signal_col in self.signal_columns:
                Y = Ys[:, index : index + available_signals[signal_col]]
                index += available_signals[signal_col]

                signal_path = (
                    self.test_signal_path.parent
                    / f"{self.quality}"
                    / "signals"
                    / f"Y_{signal_col}_{self.quality}_geneformer.csv"
                )
                pd.DataFrame(Y).to_csv(signal_path, index=False)

        return X
