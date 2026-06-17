import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import normalize_adata, select_highly_variable_genes, fit_pca, PCATransformer
from .abc import BaseAlgorithm


class PCA(BaseAlgorithm):
    def __init__(self, base_dir: str, signal_columns: list[str] | None = None, device: int = 0, seed: int = 42):
        super().__init__(base_dir, device, model_name="model", seed=seed)
        self.pca_transformer: PCATransformer | None = None
        self.signal_columns: list[str] = signal_columns if signal_columns is not None else ["celltype.l3"]
        self.hvg_mask_path = Path(base_dir).parent.parent / "utils" / "pca_hvg.pkl"

    def train(
        self,
        n_components: int = 16,
        sample_size: str = "10%",
    ) -> PCATransformer:
        sampled_adata: ad.AnnData = ad.read_h5ad(self.train_data_path / "preprocessed.h5ad")
        sampled_adata = normalize_adata(sampled_adata)
        self.pca_transformer = fit_pca(sampled_adata, self.hvg_mask_path, n_components)

        model_save_path = self.model_path / "pca_model.pkl"
        self.pca_transformer.save(model_save_path)

        return self.pca_transformer

    def embed_pca(self, adata: ad.AnnData, save_path: Path) -> np.ndarray:
        if self.pca_transformer is None:
            model_save_path = self.model_path / "pca_model.pkl"
            if not model_save_path.exists():
                raise ValueError("No trained PCA model found. Please run train() first.")
            self.pca_transformer = PCATransformer.load(model_save_path, hvg_mask_path=self.hvg_mask_path)

        adata = normalize_adata(adata)
        X_pca: np.ndarray = self.pca_transformer.transform(adata.X)
        pd.DataFrame(X_pca).to_csv(save_path, index=False)
        return X_pca

    def embed(self) -> np.ndarray:
        test_data = ad.read_h5ad(self.test_data_path / "preprocessed.h5ad")
        X_pca: np.ndarray = self.embed_pca(test_data, self.embeddings_path)

        return X_pca
