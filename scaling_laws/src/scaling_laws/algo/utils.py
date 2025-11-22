import pickle
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
from ..h5ad_reader import H5adReader
import torch


def normalize_adata(adata: ad.AnnData) -> ad.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def select_highly_variable_genes(adata: ad.AnnData) -> ad.AnnData:
    sc.pp.highly_variable_genes(adata, n_top_genes=750)
    return adata


def fit_pca(adata: ad.AnnData, path_to_hvg_mask: Path, n_components: int, random_state: int = 0x5EED):
    with open(path_to_hvg_mask, "rb") as f:
        hvg_mask = pickle.load(f)
    X = adata.X[:, hvg_mask]
    result = TruncatedSVD(n_components=n_components, random_state=random_state)
    result.fit(X)
    return PCATransformer(result, hvg_mask)


def save_results(
    save_path: Path,
    X_val_proj: np.ndarray,
    mi: float,
    embeddings: np.ndarray,
    model: torch.nn.Module,
    validation_data: ad.AnnData,
) -> None:
    lmi_results = save_path / "lmi_results"
    lmi_results.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_val_proj).to_csv(lmi_results / "validation_projected.csv", index=False)

    obs = validation_data.obs
    for col in ["development_day", "author_cell_type", "donor_id", "sex", "development_stage"]:
        if col in obs:
            pd.DataFrame(obs[col]).to_csv(lmi_results / f"Y_{col}.csv", index=False)

    with open(lmi_results / "mutual_information.txt", "w") as f:
        f.write(f"{mi:.5f}")
    np.save(lmi_results / "embeddings.npy", embeddings)
    torch.save(model.state_dict(), lmi_results / "model.pt")


class PCATransformer:
    def __init__(self, truncated_svd: TruncatedSVD, hvg_mask):
        self.truncated_svd = truncated_svd
        self.hvg_mask = hvg_mask

    def transform(self, X):
        return self.truncated_svd.transform(X[:, self.hvg_mask])

    def save(self, path: Path | str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path, hvg_mask_path: Path | None = None):
        with open(path, "rb") as f:
            transformer = pickle.load(f)

        if hvg_mask_path is not None:
            with open(hvg_mask_path, "rb") as f:
                transformer.hvg_mask = pickle.load(f)

        return transformer
