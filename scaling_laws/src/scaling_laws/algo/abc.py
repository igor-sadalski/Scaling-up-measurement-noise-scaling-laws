from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from latentmi import lmi
import glob
import os
import anndata as ad
import scvi
from geneformer import EmbExtractor
import random


class BaseAlgorithm(ABC):

    def __init__(self, base_dir: str, device: int = 0, model_name: str = "model", seed: int = 42):
        self.base_dir: Path = Path(base_dir)
        self.seed: int = seed
        self.model_name: str = model_name

        self._set_seed(seed)

        self.train_data_path: Path = self.base_dir / "preprocessed"
        self.quality = self.base_dir.name
        self.size = self.base_dir.parent.name
        self.test_data_path: Path = self.base_dir.parent.parent / "test" / self.quality / "preprocessed"
        self.validation_data_path: Path = self.base_dir.parent.parent / "validation" / self.quality / "preprocessed"
        self.test_signal_path: Path = self.base_dir.parent.parent / "test" / "signals"
        self.validation_signal_path: Path = self.base_dir.parent.parent / "validation" / "signals"
        self.dataset_name: str = self.base_dir.parent.parent.name

        self.method_name: str = self.__class__.__name__
        self.save_folder_path: Path = self.base_dir / "results" / self.method_name
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

        self.model_path: Path = self.save_folder_path / self.model_name
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.embeddings_path: Path = self.save_folder_path / self.model_name / "embeddings.csv"

        self.device: int = device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        self.vae: scvi.model.SCVI | None = None
        self.signal_columns: list[str] = ["celltype.l3"]
        self.per_device_eval_bs: int = 100
        self.embed_dim: int = 256
        self.token_dictionary_path: Path = self.base_dir.parent.parent / "utils" / "token_dict.pkl"
        self.geneformer_test_path: Path = self.test_data_path / "tokenized.dataset"

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def train(self, **kwargs) -> object:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def embed(self, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def mutual_information(self, max_epochs: int = 300, device: str = "cuda:0") -> dict:

        mi_results = {}
        quality_signal_path = self.base_dir.parent.parent / "test" / self.quality / "signals"
        signal_files = sorted(quality_signal_path.glob("*.csv"))
        print(f"Found {len(signal_files)} quality-specific signal files: {[f.name for f in signal_files]}")

        for signal_file in signal_files:

            embeddings_df = pd.read_csv(self.embeddings_path)
            embeddings: np.ndarray = embeddings_df.values
            X: np.ndarray = embeddings.astype(np.float64)

            if self.method_name.lower() == "geneformer" and not signal_file.stem.endswith("_geneformer"):
                continue
            elif self.method_name.lower() != "geneformer" and signal_file.stem.endswith("_geneformer"):
                continue
            print("using signal file", signal_file)

            signal_folder = self.save_folder_path / self.model_path.name / "MI" / str(self.seed) / signal_file.stem
            signal_folder.mkdir(parents=True, exist_ok=True)
            signal_data: np.ndarray = pd.read_csv(signal_file).values

            if self.dataset_name.lower() == "merfish":

                if "neighbours" in signal_file.stem:
                    continue

                signal_data: np.ndarray = pd.read_csv(signal_file).astype(str).values
                cur_idx, ng_idx = signal_data[:, 0], signal_data[:, 1]
                preprocessed_path = (
                    self.base_dir.parent.parent / "test" / str(self.quality) / "preprocessed" / "preprocessed.h5ad"
                )
                adata = ad.read_h5ad(preprocessed_path, backed="r")
                embeddings_df = pd.read_csv(self.embeddings_path)
                assert len(cur_idx) == len(embeddings_df), "cur_idx and embeddings_df have different lengths"
                assert set(ng_idx).issubset(set(cur_idx)), "ng_idx has values that are not in cur_idx"
                embeddings_df.index = cur_idx
                Y = embeddings_df.loc[ng_idx].values
                test_indices = adata.uns["test_indices"].astype(int)
                test_indices = test_indices[test_indices < X.shape[0]]
                X = X[test_indices].astype(np.float32)
                Y = Y[test_indices].astype(np.float32)
                assert X.shape == Y.shape, "X and Y have different shapes"

            elif self.dataset_name.lower() == "larry":

                if "clone" not in signal_file.stem:
                    continue

                signal_data: pd.DataFrame = pd.read_csv(signal_file, dtype={0: str, 1: str, 2: int})
                signal_data.columns = ["index", "clone", "time"]

                early_clones = signal_data[signal_data["time"].isin([2, 4])]
                late_clones = signal_data[signal_data["time"].isin([6])]

                common_clones = (
                    set(late_clones["clone"].values)
                    .intersection(set(early_clones["clone"].values))
                    .intersection(set(signal_data["index"].values))
                )
                assert len(common_clones) > 0, "No common clones found"

                early_common = early_clones[early_clones["clone"].isin(common_clones)].groupby("clone").sample(n=1)
                late_common = late_clones[late_clones["clone"].isin(common_clones)].groupby("clone").sample(n=1)

                embeddings_df.index = signal_data["index"].values
                X = embeddings_df.loc[early_common["index"].tolist()].values.astype(np.float32)
                Y = embeddings_df.loc[late_common["index"].tolist()].values.astype(np.float32)
                assert X.shape == Y.shape
                assert not np.allclose(X, Y), "X and Y are identical, but they should be different"
            elif signal_data.shape[1] == 1:
                signal_data: np.ndarray = pd.read_csv(signal_file).values
                Y = pd.get_dummies(signal_data.ravel()).values
            elif signal_data.shape[1] != 1:
                signal_data: np.ndarray = pd.read_csv(signal_file).values
                Y: np.ndarray = signal_data

            pmi, lmi_embeddings, model = lmi.estimate(
                X,
                Y,
                validation_split=0.3,
                batch_size=512,
                epochs=max_epochs,
                quiet=False,
            )
            mi = np.nanmean(pmi)

            print(
                f"\033[92m{self.method_name} MI for {signal_file.stem} at {self.base_dir.parent.parent.name} quality {self.quality} size {self.size}: {mi}\033[0m"
            )
            print(f"Saving results to {signal_folder}")
            mi_results[signal_file.name] = mi

            with open(signal_folder / "lmi_mutual_information.txt", "w") as f:
                f.write(f"{mi:.5f}")
            np.save(signal_folder / "lmi_embeddings.npy", lmi_embeddings)
            torch.save(model.state_dict(), signal_folder / "lmi_model.pt")

        return mi_results
