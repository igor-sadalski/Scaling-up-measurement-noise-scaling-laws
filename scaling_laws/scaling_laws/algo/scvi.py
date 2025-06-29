# bin/python
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import logging
import scvi
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from latentmi import lmi
from .abc import BaseAlgorithm


class SCVI(BaseAlgorithm):
    def __init__(
        self,
        base_dir: str,
        signal_columns: list[str] | None = None,
        device: int = 0,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 100,
        dataset_name: str | None = None,
        seed: int = 42,
    ):
        super().__init__(base_dir, device, model_name="model", seed=seed)
        self.vae: scvi.model.SCVI | None = None
        self.signal_columns: list[str] = signal_columns if signal_columns is not None else ["celltype.l3"]
        self.max_epochs: int = max_epochs
        self.early_stopping_patience: int = early_stopping_patience
        self.early_stopping_threshold: float = early_stopping_threshold
        self.dataset_name: str = dataset_name or "unknown"

    def train(
        self,
    ) -> scvi.model.SCVI:
        wandb.init(
            project="scvi-scaling-laws",
            name=f"{self.dataset_name}_size{self.size}_quality{self.quality}",
            config={
                "dataset": self.dataset_name,
                "size": self.size,
                "quality": self.quality,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            },
        )

        adata: ad.AnnData = ad.read_h5ad(
            self.train_data_path / "preprocessed.h5ad",
            backed="r",
        )

        scvi.model.SCVI.setup_anndata(adata)
        self.vae = scvi.model.SCVI(
            adata,
            n_hidden=512,
            n_latent=16,
            n_layers=1,
            dropout_rate=0.1,
            dispersion="gene",
            gene_likelihood="zinb",
            latent_distribution="normal",
        )

        wandb_logger = WandbLogger(
            project="scvi-scaling-laws",
            name=f"{self.dataset_name}_size{self.size}_quality{self.quality}",
            save_dir=str(self.save_folder_path),
            log_model=False,
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

        self.vae.train(
            accelerator="gpu",
            devices=1,
            train_size=0.8,
            validation_size=0.2,
            shuffle_set_split=True,
            load_sparse_tensor=False,
            batch_size=512,
            plan_kwargs=train_plan,
            check_val_every_n_epoch=1,
            log_every_n_steps=200,
            enable_progress_bar=True,
            simple_progress_bar=False,
            early_stopping=True,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=0.01,
            early_stopping_monitor="elbo_validation",
            max_epochs=self.max_epochs,
            early_stopping_mode="min",
            logger=wandb_logger,
        )

        self.vae.save(self.model_path, overwrite=True, save_anndata=False)

        wandb.finish()

        return self.vae

    def embed(self) -> np.ndarray:
        adata_test: ad.AnnData = ad.read_h5ad(
            self.test_data_path / "preprocessed.h5ad",
            backed="r",
        )
        print(self.test_data_path / "preprocessed.h5ad")
        self.vae = scvi.model.SCVI.load(dir_path=self.model_path, adata=adata_test)
        latent_representation = self.vae.get_latent_representation(adata_test)
        latent_df = pd.DataFrame(latent_representation)
        latent_df.to_csv(self.embeddings_path, index=False)

        return latent_representation
