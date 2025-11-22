import numpy as np
import pandas as pd
import anndata as ad
import logging
import torch
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from latentmi import lmi
import joblib
from .abc import BaseAlgorithm


class RandomProjection(BaseAlgorithm):
    def __init__(self, base_dir: str, seed: int = 42, device: int = 0):
        super().__init__(base_dir, device)
        self.seed: int = seed
        self.rp: GaussianRandomProjection | None = None

    def train(self, n_components: int = 16) -> GaussianRandomProjection:
        train_adata: ad.AnnData = ad.read_h5ad(self.train_data_path / "preprocessed.h5ad", backed="r")

        self.rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        self.rp.fit(train_adata.X[:5, :].toarray())

        joblib.dump(self.rp, self.model_path / "random_projection.joblib")

        return self.rp

    def embed(self) -> np.ndarray:
        test_data: ad.AnnData = ad.read_h5ad(self.test_data_path / "preprocessed.h5ad")

        self.rp = joblib.load(self.model_path / "random_projection.joblib")

        X_projected = self.rp.transform(test_data.X[:, :].toarray())

        pd.DataFrame(X_projected).to_csv(self.embeddings_path, index=False)

        return X_projected
