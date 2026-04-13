"""
Retrieve artifacts from the Measurement Noise Scaling Laws dataset.

The dataset is publicly available at:
    s3://measurement-noise-scaling-laws/data/

This module works with both local directories and S3 paths. For S3 access,
files are downloaded on demand into a local cache directory.

Usage:
    from scaling_laws.s3_retriever import S3Retriever

    # From S3 (public, no credentials needed)
    data = S3Retriever("s3://measurement-noise-scaling-laws/data/")

    # From a local copy
    data = S3Retriever("/path/to/local/data/")

    # Load embeddings
    emb = data.load_embeddings("merfish", num_cells=7113, quality=1.0, algorithm="Geneformer")

    # Load MI score
    mi = data.load_mutual_information("merfish", num_cells=7113, quality=1.0,
                                       algorithm="Geneformer", signal="ng_idx", seed=42)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

S3_BUCKET = "s3://measurement-noise-scaling-laws/data/"

DATASETS = ["PBMC", "larry", "merfish", "shendure"]

ALGORITHMS = ["Geneformer", "SCVI", "State", "PCA", "RandomProjection"]

SEEDS = [42, 1404, 2303, 2701]

DATASET_SIGNALS: dict[str, list[str]] = {
    "PBMC": ["celltype.l3", "protein_counts"],
    "larry": ["clone"],
    "merfish": ["cur_idx", "ng_idx"],
    "shendure": ["author_day"],
}

DATASET_SIZES: dict[str, list[int]] = {
    "PBMC": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
    "larry": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
    "merfish": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
    "shendure": [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
}

DATASET_QUALITIES: dict[str, list[float]] = {
    "PBMC": [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422,
             0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
    "larry": [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311,
              0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
    "merfish": [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267,
                0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
    "shendure": [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384,
                 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
}


@dataclass
class EmbeddingResult:
    """A single embedding configuration and its data.

    Attributes
    ----------
    dataset : str
        Dataset name (e.g. ``'merfish'``).
    num_cells : int
        Training subset size.
    quality : float
        Noise quality level (1.0 = no noise).
    algorithm : str
        Embedding algorithm name.
    signals : list[str]
        Biological signal columns available for this dataset.
    embedding : pd.DataFrame
        Embedding matrix (cells x dimensions).
    embedding_dim : int
        Number of embedding dimensions.
    """

    dataset: str
    num_cells: int
    quality: float
    algorithm: str
    signals: list[str]
    embedding: pd.DataFrame
    embedding_dim: int


@dataclass
class EmbeddingSignalResult:
    """A matched (embedding, signal) pair for MI recomputation.

    Rows in ``embedding`` and ``signal_data`` are aligned: row *i* in the
    embedding corresponds to row *i* in the signal DataFrame.

    Attributes
    ----------
    dataset : str
        Dataset name.
    num_cells : int
        Training subset size.
    quality : float
        Noise quality level (1.0 = no noise).
    algorithm : str
        Embedding algorithm name.
    signal : str
        Biological signal column name (e.g. ``'ng_idx'``, ``'celltype.l3'``).
    embedding : pd.DataFrame
        Embedding matrix (cells x dimensions).
    embedding_dim : int
        Number of embedding dimensions.
    signal_data : pd.DataFrame
        Signal data loaded from the test signal CSV, row-aligned with
        ``embedding``.
    """

    dataset: str
    num_cells: int
    quality: float
    algorithm: str
    signal: str
    embedding: pd.DataFrame
    embedding_dim: int
    signal_data: pd.DataFrame


class S3Retriever:
    """Retrieve artifacts from the Measurement Noise Scaling Laws dataset.

    Works transparently with local paths and S3 URIs. When pointed at S3,
    individual files are downloaded on first access and cached locally.

    Parameters
    ----------
    root : str
        Root path to the data directory. Can be a local path or an S3 URI
        (e.g. ``s3://measurement-noise-scaling-laws/data/``).
    cache_dir : str or None
        Local directory for caching files downloaded from S3. Defaults to
        ``~/.cache/noise_scaling_laws/``. Ignored when *root* is a local path.
    """

    def __init__(self, root: str = S3_BUCKET, cache_dir: str | None = None):
        self._is_s3 = root.startswith("s3://")
        self._root = root.rstrip("/")
        if self._is_s3:
            self._cache = Path(cache_dir or os.path.expanduser("~/.cache/noise_scaling_laws"))
            self._cache.mkdir(parents=True, exist_ok=True)
        else:
            self._root_path = Path(root)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve(self, *parts: str) -> Path:
        """Return a local Path to the requested file, downloading from S3 if necessary."""
        rel = "/".join(str(p) for p in parts)
        if not self._is_s3:
            return self._root_path / rel

        local = self._cache / rel
        if not local.exists():
            s3_uri = f"{self._root}/{rel}"
            local.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "cp", s3_uri, str(local), "--no-sign-request"],
                check=True,
                capture_output=True,
            )
        return local

    def _sync_s3_dir(self, rel: str) -> Path:
        """Sync an S3 directory to local cache. Returns the local path."""
        local_dir = self._cache / rel
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync", f"{self._root}/{rel}/", str(local_dir), "--no-sign-request"],
                check=True,
                capture_output=True,
            )
        return local_dir

    def _experiment_dir(self, dataset: str, num_cells: int, quality: float) -> str:
        """Return the relative path to an experiment directory."""
        return f"{dataset}/{num_cells}/{quality}"

    def _results_dir(self, dataset: str, num_cells: int, quality: float, algorithm: str) -> str:
        return f"{self._experiment_dir(dataset, num_cells, quality)}/results/{algorithm}"

    def _model_dir(self, dataset: str, num_cells: int, quality: float, algorithm: str,
                   model_name: str = "model") -> str:
        return f"{self._results_dir(dataset, num_cells, quality, algorithm)}/{model_name}"

    def _signal_path(self, dataset: str, quality: float, signal: str, algorithm: str) -> str:
        """Return the relative path to a test signal CSV.

        Geneformer uses its own signal file variant with a ``_geneformer``
        suffix.  For other algorithms the standard file is used.
        """
        q = str(quality)
        if algorithm == "Geneformer":
            return f"{dataset}/test/{q}/signals/Y_{signal}_{q}_geneformer.csv"
        return f"{dataset}/test/{q}/signals/Y_{signal}_{q}.csv"

    def _mi_dir_candidates(self, algorithm: str, quality: float, signal: str, seed: int) -> list[str]:
        """Return candidate MI sub-paths to try (naming conventions vary by algorithm)."""
        method_suffix = algorithm.lower()
        if algorithm == "RandomProjection":
            method_suffix = "randomprojection"
        elif algorithm == "State":
            method_suffix = ""

        q = str(quality)
        candidates = [f"MI/{seed}/Y_{signal}_{q}_{method_suffix}"]
        # State and some others omit the method suffix
        candidates.append(f"MI/{seed}/Y_{signal}_{q}")
        return candidates

    # ------------------------------------------------------------------
    # Metadata / discovery
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[str]:
        """Return the names of available datasets."""
        return list(DATASETS)

    def list_sizes(self, dataset: str) -> list[int]:
        """Return available sampling sizes for a dataset."""
        return DATASET_SIZES[dataset]

    def list_qualities(self, dataset: str) -> list[float]:
        """Return available quality levels for a dataset."""
        return DATASET_QUALITIES[dataset]

    def list_algorithms(self, dataset: str | None = None) -> list[str]:
        """Return available algorithms. All datasets share the same set,
        though not every (dataset, size, quality) combination has all five."""
        return list(ALGORITHMS)

    def list_signals(self, dataset: str) -> list[str]:
        """Return the biological signal columns measured for a dataset."""
        return DATASET_SIGNALS[dataset]

    def list_experiments(self, dataset: str) -> pd.DataFrame:
        """Return a DataFrame listing all (size, quality) combinations for a dataset."""
        rows = [
            {"dataset": dataset, "num_cells": s, "quality": q}
            for s in self.list_sizes(dataset)
            for q in self.list_qualities(dataset)
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Raw / preprocessed data
    # ------------------------------------------------------------------

    def load_raw(self, dataset: str) -> "anndata.AnnData":
        """Load the raw (unprocessed) AnnData object for a dataset.

        Returns an AnnData backed in read mode to avoid loading the full
        object into memory for large datasets.
        """
        import anndata
        path = self._resolve(dataset, "raw", "raw.h5ad")
        return anndata.read_h5ad(path, backed="r")

    def load_sample(self, dataset: str, num_cells: int, quality: float) -> "anndata.AnnData":
        """Load the preprocessed AnnData for a specific (size, quality) combination."""
        import anndata
        path = self._resolve(
            dataset, str(num_cells), str(quality),
            "preprocessed", "preprocessed.h5ad",
        )
        return anndata.read_h5ad(path)

    def load_test_sample(self, dataset: str, quality: float) -> "anndata.AnnData":
        """Load the preprocessed test set at a given quality level."""
        import anndata
        path = self._resolve(dataset, "test", str(quality), "preprocessed", "preprocessed.h5ad")
        return anndata.read_h5ad(path)

    def load_test_signal(
        self,
        dataset: str,
        quality: float,
        signal: str,
        algorithm: str,
    ) -> pd.DataFrame:
        """Load the test-set signal CSV for a given (dataset, quality, signal, algorithm).

        Geneformer uses its own signal file variant; all other algorithms
        share the standard file.

        Parameters
        ----------
        dataset : str
            Dataset name.
        quality : float
            Noise quality level.
        signal : str
            Signal column name (e.g. ``'ng_idx'``, ``'celltype.l3'``).
        algorithm : str
            Embedding algorithm — determines which signal file variant to load.

        Returns
        -------
        pd.DataFrame
            Signal data with one row per test cell (row-aligned with the
            embedding from :meth:`load_embeddings`).
        """
        path = self._resolve(self._signal_path(dataset, quality, signal, algorithm))
        return pd.read_csv(path)

    def load_tokenized_dataset(self, dataset: str, num_cells: int, quality: float) -> "datasets.Dataset":
        """Load the tokenized HuggingFace dataset (used by Geneformer)."""
        from datasets import load_from_disk
        rel = f"{dataset}/{num_cells}/{quality}/preprocessed/tokenized.dataset"
        if self._is_s3:
            local_dir = self._sync_s3_dir(rel)
            return load_from_disk(str(local_dir))
        return load_from_disk(str(self._root_path / rel))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_embeddings(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        algorithm: str,
        model_name: str = "model",
    ) -> pd.DataFrame:
        """Load test-set embeddings (CSV) for a trained model.

        Parameters
        ----------
        dataset : str
            One of 'PBMC', 'larry', 'merfish', 'shendure'.
        num_cells : int
            Training subset size.
        quality : float
            Noise quality level (1.0 = no noise).
        algorithm : str
            One of 'Geneformer', 'SCVI', 'State', 'PCA', 'RandomProjection'.
        model_name : str
            Sub-directory under the algorithm results (default ``"model"``).
            Pass ``"checkpoint-3000"`` etc. to load checkpoint embeddings.

        Returns
        -------
        pd.DataFrame
            Embedding matrix with cells as rows and dimensions as columns.
        """
        path = self._resolve(
            self._model_dir(dataset, num_cells, quality, algorithm, model_name),
            "embeddings.csv",
        )
        return pd.read_csv(path, index_col=0)

    def load_embeddings_numpy(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        algorithm: str,
        model_name: str = "model",
    ) -> np.ndarray:
        """Same as :meth:`load_embeddings` but returns a NumPy array."""
        return self.load_embeddings(dataset, num_cells, quality, algorithm, model_name).values

    def iter_embeddings(
        self,
        datasets: list[str] | None = None,
        sizes: dict[str, list[int]] | None = None,
        qualities: dict[str, list[float]] | None = None,
        algorithms: list[str] | None = None,
        model_name: str = "model",
    ) -> Iterator[EmbeddingResult]:
        """Iterate over embedding configurations, yielding an :class:`EmbeddingResult` for each.

        Skips configurations whose embedding file is missing.

        Parameters
        ----------
        datasets : list[str] or None
            Datasets to include (default: all).
        sizes : dict[str, list[int]] or None
            Per-dataset sizes to include. If *None*, all sizes for each dataset
            are used. Keys that don't appear fall back to the full list.
        qualities : dict[str, list[float]] or None
            Per-dataset quality levels to include (same logic as *sizes*).
        algorithms : list[str] or None
            Algorithms to include (default: all).
        model_name : str
            Model sub-directory (default ``"model"``).

        Yields
        ------
        EmbeddingResult
            One result per successfully loaded (dataset, size, quality, algorithm)
            combination.
        """
        datasets = datasets or DATASETS
        algorithms = algorithms or ALGORITHMS

        # Pre-compute total number of configurations for the progress bar.
        configs = []
        for ds in datasets:
            ds_sizes = (sizes or {}).get(ds, self.list_sizes(ds))
            ds_quals = (qualities or {}).get(ds, self.list_qualities(ds))
            for n in ds_sizes:
                for q in ds_quals:
                    for algo in algorithms:
                        configs.append((ds, n, q, algo))

        pbar = tqdm(configs, desc="Loading embeddings", unit="config")
        for ds, n, q, algo in pbar:
            pbar.set_description(f"{ds} | n={n} | q={q} | {algo}")
            try:
                emb = self.load_embeddings(ds, n, q, algo, model_name)
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            yield EmbeddingResult(
                dataset=ds,
                num_cells=n,
                quality=q,
                algorithm=algo,
                signals=self.list_signals(ds),
                embedding=emb,
                embedding_dim=emb.shape[1],
            )

    def iter_embeddings_with_signals(
        self,
        datasets: list[str] | None = None,
        sizes: dict[str, list[int]] | None = None,
        qualities: dict[str, list[float]] | None = None,
        algorithms: list[str] | None = None,
        signals: dict[str, list[str]] | None = None,
        model_name: str = "model",
    ) -> Iterator[EmbeddingSignalResult]:
        """Iterate over (embedding, signal) pairs for MI recomputation.

        For each (dataset, size, quality, algorithm, signal) combination,
        yields an :class:`EmbeddingSignalResult` containing the row-aligned
        embedding and signal DataFrames.  Configurations whose embedding or
        signal file is missing are silently skipped.

        Parameters
        ----------
        datasets : list[str] or None
            Datasets to include (default: all).
        sizes : dict[str, list[int]] or None
            Per-dataset sizes to include (default: all).
        qualities : dict[str, list[float]] or None
            Per-dataset quality levels to include (default: all).
        algorithms : list[str] or None
            Algorithms to include (default: all).
        signals : dict[str, list[str]] or None
            Per-dataset signal columns to include.  If *None*, all signals
            for each dataset are used.
        model_name : str
            Model sub-directory (default ``"model"``).

        Yields
        ------
        EmbeddingSignalResult
            One result per successfully loaded (dataset, size, quality,
            algorithm, signal) combination.
        """
        ds_list = datasets or DATASETS
        algo_list = algorithms or ALGORITHMS

        configs: list[tuple[str, int, float, str, str]] = []
        for ds in ds_list:
            ds_sizes = (sizes or {}).get(ds, self.list_sizes(ds))
            ds_quals = (qualities or {}).get(ds, self.list_qualities(ds))
            ds_signals = (signals or {}).get(ds, self.list_signals(ds))
            for n in ds_sizes:
                for q in ds_quals:
                    for algo in algo_list:
                        for sig in ds_signals:
                            configs.append((ds, n, q, algo, sig))

        pbar = tqdm(configs, desc="Loading embeddings + signals", unit="config")
        for ds, n, q, algo, sig in pbar:
            pbar.set_description(f"{ds} | n={n} | q={q} | {algo} | {sig}")
            try:
                emb = self.load_embeddings(ds, n, q, algo, model_name)
                sig_df = self.load_test_signal(ds, q, sig, algo)
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            yield EmbeddingSignalResult(
                dataset=ds,
                num_cells=n,
                quality=q,
                algorithm=algo,
                signal=sig,
                embedding=emb,
                embedding_dim=emb.shape[1],
                signal_data=sig_df,
            )

    # ------------------------------------------------------------------
    # Trained models
    # ------------------------------------------------------------------

    def load_geneformer_model(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        model_name: str = "model",
    ) -> Path:
        """Load a fine-tuned Geneformer model (HuggingFace BertForMaskedLM).

        Returns the model directory path so it can be loaded with
        ``transformers.AutoModel.from_pretrained(path)``.
        """
        model_dir = self._model_dir(dataset, num_cells, quality, "Geneformer", model_name)
        for fname in ("config.json", "model.safetensors"):
            self._resolve(model_dir, fname)
        if self._is_s3:
            return self._cache / model_dir
        return self._root_path / model_dir

    def load_scvi_model(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
    ) -> Path:
        """Return the local path to the scVI model.pt file.

        Load with ``torch.load(path)`` or via scvi-tools model loading.
        """
        return self._resolve(
            self._model_dir(dataset, num_cells, quality, "SCVI"),
            "model.pt",
        )

    def load_pca_model(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
    ):
        """Load the fitted PCA model (scikit-learn PCA object)."""
        import joblib
        path = self._resolve(
            self._model_dir(dataset, num_cells, quality, "PCA"),
            "pca_model.pkl",
        )
        return joblib.load(path)

    def load_random_projection_model(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
    ):
        """Load the fitted random projection model."""
        import joblib
        path = self._resolve(
            self._model_dir(dataset, num_cells, quality, "RandomProjection"),
            "random_projection.joblib",
        )
        return joblib.load(path)

    def load_state_checkpoint(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
    ) -> Path:
        """Return the local path to the STATE model checkpoint directory."""
        model_dir = self._model_dir(dataset, num_cells, quality, "State")
        if self._is_s3:
            return self._sync_s3_dir(f"{model_dir}/checkpoints")
        return self._root_path / model_dir / "checkpoints"

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    def load_mutual_information(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        algorithm: str,
        signal: str,
        seed: int = 42,
    ) -> float:
        """Load the MI estimate (scalar, in nats) for a single experiment.

        Parameters
        ----------
        signal : str
            Biological signal column (e.g. ``'ng_idx'``, ``'celltype.l3'``).
        seed : int
            Random seed used for the LMI estimator.
        """
        model_dir = self._model_dir(dataset, num_cells, quality, algorithm)
        for candidate in self._mi_dir_candidates(algorithm, quality, signal, seed):
            try:
                path = self._resolve(model_dir, candidate, "lmi_mutual_information.txt")
                return float(path.read_text().strip())
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        raise FileNotFoundError(
            f"MI result not found for {dataset}/{num_cells}/{quality}/{algorithm} "
            f"signal={signal} seed={seed}"
        )

    def load_lmi_embeddings(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        algorithm: str,
        signal: str,
        seed: int = 42,
    ) -> np.ndarray:
        """Load the LMI-optimized embedding projection."""
        model_dir = self._model_dir(dataset, num_cells, quality, algorithm)
        for candidate in self._mi_dir_candidates(algorithm, quality, signal, seed):
            try:
                path = self._resolve(model_dir, candidate, "lmi_embeddings.npy")
                return np.load(path)
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        raise FileNotFoundError(
            f"LMI embeddings not found for {dataset}/{num_cells}/{quality}/{algorithm} "
            f"signal={signal} seed={seed}"
        )

    def load_lmi_model(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        algorithm: str,
        signal: str,
        seed: int = 42,
    ) -> Path:
        """Return the local path to the trained LMI model (.pt file)."""
        model_dir = self._model_dir(dataset, num_cells, quality, algorithm)
        for candidate in self._mi_dir_candidates(algorithm, quality, signal, seed):
            try:
                return self._resolve(model_dir, candidate, "lmi_model.pt")
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        raise FileNotFoundError(
            f"LMI model not found for {dataset}/{num_cells}/{quality}/{algorithm} "
            f"signal={signal} seed={seed}"
        )

    # ------------------------------------------------------------------
    # Bulk collection
    # ------------------------------------------------------------------

    def collect_all_mi_results(
        self,
        dataset: str,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """Collect all MI results for a dataset into a single DataFrame.

        Parameters
        ----------
        dataset : str
            Dataset name.
        algorithms : list[str] or None
            Algorithms to include (default: all).
        seeds : list[int] or None
            Seeds to include (default: all four).

        Returns
        -------
        pd.DataFrame
            Columns: dataset, num_cells, quality, algorithm, signal, seed, mi
        """
        if algorithms is None:
            algorithms = ALGORITHMS
        if seeds is None:
            seeds = SEEDS

        rows = []
        for num_cells in self.list_sizes(dataset):
            for quality in self.list_qualities(dataset):
                for algo in algorithms:
                    for signal in self.list_signals(dataset):
                        for seed in seeds:
                            try:
                                mi = self.load_mutual_information(
                                    dataset, num_cells, quality, algo, signal, seed
                                )
                                rows.append({
                                    "dataset": dataset,
                                    "num_cells": num_cells,
                                    "quality": quality,
                                    "algorithm": algo,
                                    "signal": signal,
                                    "seed": seed,
                                    "mi": mi,
                                })
                            except FileNotFoundError:
                                continue

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_utils(self, dataset: str) -> dict[str, Path]:
        """Return local paths to all utility files for a dataset."""
        util_files = [
            "token_dict.pkl",
            "gene_median_dict.pkl",
            "detected_gene_median_dict.pkl",
            "ensembl_mapping_dict.pkl",
            "pca_hvg.pkl",
            "total_gene_tdigest_dict.pkl",
        ]
        return {
            f: self._resolve(dataset, "utils", f)
            for f in util_files
        }

    def download_dataset(self, dataset: str, local_dir: str | Path) -> None:
        """Download an entire dataset from S3 to a local directory.

        Only works when the root is an S3 URI.
        """
        if not self._is_s3:
            raise ValueError("download_dataset() is only available when root is an S3 URI")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        s3_uri = f"{self._root}/{dataset}/"
        subprocess.run(
            ["aws", "s3", "sync", s3_uri, str(local_dir / dataset), "--no-sign-request"],
            check=True,
        )

    def download_experiment(
        self,
        dataset: str,
        num_cells: int,
        quality: float,
        local_dir: str | Path,
    ) -> None:
        """Download a single (dataset, size, quality) experiment from S3."""
        if not self._is_s3:
            raise ValueError("download_experiment() is only available when root is an S3 URI")
        local_dir = Path(local_dir)
        rel = self._experiment_dir(dataset, num_cells, quality)
        target = local_dir / rel
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["aws", "s3", "sync", f"{self._root}/{rel}/", str(target), "--no-sign-request"],
            check=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download or inspect the Measurement Noise Scaling Laws dataset."
    )
    sub = parser.add_subparsers(dest="command")

    # -- list --
    p_list = sub.add_parser("list", help="List available experiments")
    p_list.add_argument("dataset", choices=DATASETS)

    # -- download --
    p_dl = sub.add_parser("download", help="Download data from S3")
    p_dl.add_argument("dataset", choices=DATASETS)
    p_dl.add_argument("--local-dir", default="./data", help="Local destination directory")
    p_dl.add_argument("--num-cells", type=int, help="Download only this size (optional)")
    p_dl.add_argument("--quality", type=float, help="Download only this quality (optional)")

    # -- mi --
    p_mi = sub.add_parser("mi", help="Print MI score for a single experiment")
    p_mi.add_argument("dataset", choices=DATASETS)
    p_mi.add_argument("--num-cells", type=int, required=True)
    p_mi.add_argument("--quality", type=float, required=True)
    p_mi.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    p_mi.add_argument("--signal", required=True)
    p_mi.add_argument("--seed", type=int, default=42)

    # -- collect --
    p_collect = sub.add_parser("collect", help="Collect all MI results into a CSV")
    p_collect.add_argument("dataset", choices=DATASETS)
    p_collect.add_argument("--output", default=None, help="Output CSV path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    root = os.environ.get("SCALING_LAWS_DATA", S3_BUCKET)

    data = S3Retriever(root)

    if args.command == "list":
        df = data.list_experiments(args.dataset)
        print(f"Dataset: {args.dataset}")
        print(f"Sizes: {data.list_sizes(args.dataset)}")
        print(f"Qualities: {data.list_qualities(args.dataset)}")
        print(f"Signals: {data.list_signals(args.dataset)}")
        print(f"Total experiments: {len(df)}")

    elif args.command == "download":
        if args.num_cells and args.quality:
            data.download_experiment(args.dataset, args.num_cells, args.quality, args.local_dir)
            print(f"Downloaded {args.dataset}/{args.num_cells}/{args.quality} to {args.local_dir}")
        else:
            data.download_dataset(args.dataset, args.local_dir)
            print(f"Downloaded {args.dataset} to {args.local_dir}")

    elif args.command == "mi":
        mi = data.load_mutual_information(
            args.dataset, args.num_cells, args.quality, args.algorithm, args.signal, args.seed
        )
        print(f"MI({args.dataset}, n={args.num_cells}, q={args.quality}, "
              f"{args.algorithm}, {args.signal}, seed={args.seed}) = {mi:.6f}")

    elif args.command == "collect":
        df = data.collect_all_mi_results(args.dataset)
        output = args.output or f"{args.dataset}_mi_results.csv"
        df.to_csv(output, index=False)
        print(f"Collected {len(df)} MI results -> {output}")
