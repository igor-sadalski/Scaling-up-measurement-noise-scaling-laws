from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import os
import pickle
import shutil
import time
import scvi
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import itertools
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import crick
import numpy as np
import anndata as ad
from datasets import load_from_disk
import numpy as np
from pathlib import Path
import pandas as pd
import anndata

import scanpy as sc
import subprocess
import scipy.sparse as sp

from geneformer import TranscriptomeTokenizer
from scaling_laws.algo import *
from pathlib import Path

from sklearn.neighbors import NearestNeighbors

from .utils import (
    compute_sums,
    downsample_matrix,
    compute_medians,
    merge_digest,
)

from multiprocessing import Pool
import random

from scaling_laws.h5ad_reader import H5adReader


class PrepareData:
    def __init__(self, base_dir: str):
        self.base_dir: Path = Path(base_dir)
        self.quality: str = self.base_dir.name

        self.preprocessed: Path = self.base_dir / "preprocessed"
        self.utils_path: Path = self.base_dir.parent.parent / "utils"
        self.raw_data_path: Path = self.base_dir.parent.parent / "raw"

        self.preprocessed.mkdir(parents=True, exist_ok=True)
        self.utils_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def tokenize(
        self,
        model_input_size: int = 512,
        signal_columns: list[str] | None = None,
        chunk_size: int = 100_000,
        nproc: int = 50,
    ):
        if signal_columns is None:
            signal_columns = []

        custom_attr_name_dict = {}
        for col in signal_columns:
            if col == "protein_counts":
                adata = ad.read_h5ad(self.preprocessed / "preprocessed.h5ad", backed="r")
                for col in adata.obs.columns:
                    if col.startswith("prot_"):
                        custom_attr_name_dict[col] = col
                del adata
                gc.collect()
            else:
                custom_attr_name_dict[col] = col

        tk = TranscriptomeTokenizer(
            model_input_size=model_input_size,
            special_token=False,
            collapse_gene_ids=True,
            gene_median_file=self.utils_path / "gene_median_dict.pkl",
            token_dictionary_file=self.utils_path / "token_dict.pkl",
            gene_mapping_file=self.utils_path / "ensembl_mapping_dict.pkl",
            custom_attr_name_dict=custom_attr_name_dict,
            chunk_size=chunk_size,
            nproc=nproc,
        )

        adata = ad.read_h5ad(self.preprocessed / "preprocessed.h5ad", backed="r")
        assert (adata.obs["n_counts"] > 0).all(), "All values in n_counts should be larger than zero"
        del adata

        tk.tokenize_data(
            data_directory=self.preprocessed,
            output_directory=self.preprocessed,
            output_prefix="tokenized",
            file_format="h5ad",
            use_generator=True,
        )

        tokenized_dataset = load_from_disk(self.preprocessed / f"tokenized.dataset")
        lengths = tokenized_dataset["length"]
        assert all(
            l != 0 for l in lengths
        ), "Found 0 in tokenized sequence lengths. Tokenization may have failed for some cells."
        lengths_path = self.preprocessed / f"lengths.pkl"
        with open(lengths_path, "wb") as f:
            pickle.dump(lengths, f, protocol=4)
        print(f"save path saving")

    def prepare_for_geneformer(self):

        print(f"prepare for {self.preprocessed}")
        adata = ad.read_h5ad(self.preprocessed / "preprocessed.h5ad")

        initial_cells = len(adata)
        adata = adata[adata.obs["n_counts"] > 0]
        removed_cells = initial_cells - len(adata)
        print(f"Removed {removed_cells} cells ({removed_cells/initial_cells:.1%} of total) with zero counts")

        adata.write_h5ad(self.preprocessed / "preprocessed.h5ad")

    def median_files(self, sizes: list[int]):

        partition_files = [self.preprocessed.parent.parent.parent / "raw" / "raw.h5ad"]

        all_genes = set()
        partition_digests = {}

        results = []
        for partition_file in partition_files:
            result = compute_medians(str(partition_file))
            partition_file, digest_dict, genes = result
            partition_digests[partition_file] = digest_dict
            all_genes.update(genes)
            results.append(result)

        all_genes = list(all_genes)

        total_digests = [crick.tdigest.TDigest() for _ in range(len(all_genes))]
        total_digest_dict = dict(zip(all_genes, total_digests))

        for partition_file, digest_dict, _ in tqdm(results, desc="Merging partitions"):
            total_digest_dict = {k: merge_digest(k, v, digest_dict) for k, v in total_digest_dict.items()}

        with open(self.utils_path / "total_gene_tdigest_dict.pkl", "wb") as fp:
            pickle.dump(total_digest_dict, fp)

        gene_median_dict = {
            k: v.quantile(0.5) if not np.isnan(v.quantile(0.5)) else 1 for k, v in tqdm(total_digest_dict.items())
        }

        with open(self.utils_path / "gene_median_dict.pkl", "wb") as fp:
            pickle.dump(gene_median_dict, fp)

        detected_median_dict = {k: v for k, v in gene_median_dict.items() if not np.isnan(v)}
        with open(self.utils_path / "detected_gene_median_dict.pkl", "wb") as fp:
            pickle.dump(detected_median_dict, fp)

        filtered_genes = [gene for gene, value in gene_median_dict.items() if not np.isnan(value)]

        token_dict = {"<pad>": 0, "<mask>": 1}
        for i, gene in enumerate(filtered_genes):
            token_dict[gene] = i + 2

        with open(self.utils_path / "token_dict.pkl", "wb") as fp:
            pickle.dump(token_dict, fp)

        print(f"Number of genes in token_dict: {len(token_dict)}")
        print(f"Number of genes in gene_median_dict: {len(gene_median_dict)}")

        ensembl_mapping_dict = {gene_id: gene_id for gene_id in gene_median_dict.keys()}
        with open(self.utils_path / "ensembl_mapping_dict.pkl", "wb") as f:
            pickle.dump(ensembl_mapping_dict, f)

    @staticmethod
    def downsample(
        ratio: float,
        path_to_sample_data: str | Path,
        output_dir: str | Path | None = None,
        chunk_size: int = 300_000,
    ):

        output_dir = Path(output_dir) if output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        adata = ad.read_h5ad(str(path_to_sample_data), backed="r")
        preprocessed_chunks = []
        n_cells = adata.n_obs

        for i in tqdm(range(0, n_cells, chunk_size), desc="Processing chunks"):
            chunk = adata[i : i + chunk_size].to_memory()

            csc_matrix = chunk.X.tocsc()

            data = csc_matrix.data
            indices = csc_matrix.indices
            indptr = csc_matrix.indptr
            n_cols = csc_matrix.shape[1]

            if ratio is not None:
                downsample_matrix(data, indices, indptr, n_cols, ratio)

            csc_matrix.eliminate_zeros()
            chunk.X = csc_matrix.tocsr()

            data, indices, indptr = chunk.X.data, chunk.X.indices, chunk.X.indptr
            n_rows, n_cols = chunk.X.shape

            n_counts = compute_sums(data, indices, indptr, n_rows, n_cols)

            chunk.obs["n_counts"] = n_counts

            original_size = len(chunk)
            chunk = chunk[chunk.obs["n_counts"] > 0]
            removed_percentage = (original_size - len(chunk)) / original_size * 100
            print(f"Removed {removed_percentage:.2f}% of cells in this chunk")

            preprocessed_chunks.append(chunk)

        adata_downsampled = ad.concat(preprocessed_chunks, join="outer")
        adata_downsampled.uns = adata.uns

        adata_downsampled.var["ensembl_id"] = adata_downsampled.var.index

        assert (adata_downsampled.obs["n_counts"] > 0).all(), "All values in n_counts should be larger than zero"

        adata_downsampled.write_h5ad(output_dir / "downsampled.h5ad")

        return adata_downsampled

    @staticmethod
    def sample(
        sample_size: int = 100_000,
        chunk_size: int = 100_000,
        path_to_dataset_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        held_out_embryos: bool = False,
        every_other_timepoint: bool = False,
    ):
        path_to_dataset_dir = Path(path_to_dataset_dir) if path_to_dataset_dir else Path(".")
        path_to_sample = path_to_dataset_dir / f"{sample_size}"
        path_to_sample.mkdir(parents=True, exist_ok=True)

        raw_h5ad_path = str(path_to_dataset_dir / "raw" / "raw.h5ad")
        with H5adReader(raw_h5ad_path, chunk_size=chunk_size) as reader:
            sample = reader.sample(sample_size=sample_size, concat=True)
            sample.write_h5ad(str(output_file) if output_file else str(path_to_sample / "sample.h5ad"))

    @staticmethod
    def download_raw_data(dataset_name: str, path_to_data_dir: str = None):

        path_to_data_dir = Path(path_to_data_dir)
        raw_data_path = path_to_data_dir / f"{dataset_name}" / "raw"
        raw_data_path.mkdir(parents=True, exist_ok=True)

        if dataset_name == "PBMC":
            adata = scvi.data.pbmc_seurat_v4_cite_seq(apply_filters=True, aggregate_proteins=True)
            protein_df = pd.DataFrame(
                adata.obsm["protein_counts"].values,
                index=adata.obs_names,
                columns=[f"prot_{i}" for i in range(adata.obsm["protein_counts"].shape[1])],
            )
            adata.obs = pd.concat([adata.obs, protein_df], axis=1)

        elif dataset_name == "shendure":
            adata = ad.read_h5ad(
                "/home/jupyter/gokul_paper/somite-models/Geneformer/data/subsamples/shendure_sample_1000000.h5ad",
                backed=True,
            )

        elif dataset_name == "merfish":
            meta = pd.read_csv("/home/jupyter/igor_repos/noise_scaling_laws/data/raw/S1R1_meta.csv", index_col=0)
            cxg = pd.read_csv("/home/jupyter/igor_repos/noise_scaling_laws/data/raw/S1R1_cxg.csv", index_col=0)
            rnas = [x for x in cxg.keys() if "Blank" not in x]
            adata = ad.AnnData(cxg[rnas])
            sparse_X = sp.csr_matrix(adata.X)
            adata.X = sparse_X
            adata.obs["center_x"] = meta["center_x"]
            adata.obs["center_y"] = meta["center_y"]

            def sample_neighbors(data, k_neighbors=5, x="center_x", y="center_y"):
                xy = data.obs[[x, y]]
                nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(xy)
                _, positions = nbrs.kneighbors(xy)
                nbs = positions[np.arange(len(xy)), np.random.randint(1, high=k_neighbors, size=len(xy))]
                unique_indexes = data.obs.index[nbs]
                assert len(unique_indexes) == len(data.obs.index.unique()), "Some indexes are not unique"
                return unique_indexes

            adata.obs["ng_idx"] = sample_neighbors(adata, k_neighbors=2).values
            adata.obs["cur_idx"] = adata.obs.index.values

        elif dataset_name == "larry":
            urls = {
                "counts": "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_normed_counts.mtx.gz",
                "meta": "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_metadata.txt.gz",
                "genes": "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_gene_names.txt.gz",
                "clones": "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_clone_matrix.mtx.gz",
            }
            root = raw_data_path / "larry"
            root.mkdir(parents=True, exist_ok=True)

            for name, url in urls.items():
                fn = root / Path(url).name
                if not fn.with_suffix("").exists():
                    subprocess.run(["wget", url], cwd=root, check=True)
                    subprocess.run(["gzip", "-d", fn.name], cwd=root, check=True)

            counts: ad.AnnData = sc.read_mtx(root / "stateFate_inVitro_normed_counts.mtx")
            clones: ad.AnnData = sc.read_mtx(root / "stateFate_inVitro_clone_matrix.mtx")
            meta: pd.DataFrame = pd.read_csv(root / "stateFate_inVitro_metadata.txt", sep="\t")
            genes: pd.DataFrame = pd.read_csv(root / "stateFate_inVitro_gene_names.txt", header=None, sep="\t")

            counts.var_names = [g.upper() for g in genes[0].values]
            clone_ids = [int(np.argmax(clones.X[i, :])) for i in range(clones.shape[0])]

            counts.obs["index"] = counts.obs.index.astype(str)
            indexes = counts.obs["index"].apply(lambda x: str(abs(hash(x))))
            counts.obs["index"] = indexes
            clones = pd.DataFrame(counts.obs["index"]).iloc[clone_ids].values
            counts.obs["clone"] = clones.astype(str)
            counts.obs["time"] = meta["Time point"].values
            counts.obs["well"] = meta["Well"].values
            counts.obs["type"] = meta["Cell type annotation"].values
            counts.obs["SPRING1"] = meta["SPRING-x"].values
            counts.obs["SPRING2"] = meta["SPRING-y"].values

            def convert_to_raw_counts(sparse_matrix):
                min_per_row = np.full(sparse_matrix.shape[0], np.inf)
                for i in range(sparse_matrix.shape[0]):
                    row_data = sparse_matrix.getrow(i).data
                    if row_data.size > 0:
                        min_per_row[i] = np.min(row_data)
                return min_per_row

            nonzero_min_per_row = convert_to_raw_counts(counts.X)
            reciprocal_min_per_row = 1 / nonzero_min_per_row
            diagonal_matrix = sp.diags(reciprocal_min_per_row)
            result_matrix = diagonal_matrix.dot(counts.X)
            counts.X = result_matrix.astype(int)
            adata = counts
        else:
            raise ValueError(f"Dataset name '{dataset_name}' not found.")

        chunk_size = 100_000
        counts = []
        for i in tqdm(range(0, adata.shape[0], chunk_size), desc="Processing chunk rows for n_counts"):
            chunk_part = adata[i : i + chunk_size]
            data, indices, indptr = chunk_part.X.data, chunk_part.X.indices, chunk_part.X.indptr
            n_rows, n_cols = chunk_part.X.shape

            n_counts = compute_sums(data, indices, indptr, n_rows, n_cols)

            counts.append(n_counts)

        adata.obs["n_counts"] = np.concatenate(counts)
        adata.var["ensembl_id"] = adata.var.index.str.upper()
        adata.var.index = adata.var.index.str.upper()
        adata.write_h5ad(raw_data_path / "raw.h5ad")

    def create_training_set(
        self,
    ):
        downsampled_path = self.base_dir / "downsampled.h5ad"
        if downsampled_path.exists():
            train_path = self.preprocessed / "preprocessed.h5ad"
            downsampled_path.rename(train_path)
        else:
            raise FileNotFoundError(f"Downsampled data not found at {downsampled_path}")


class ExperimentJobIterator:
    """Iterator that yields experiment job arguments with GPU allocation handled automatically."""

    def __init__(
        self,
        datasets,
        sizes,
        qualities,
        algos,
        configs,
        path_to_data_dir,
        signal_columns,
        sleep_time=30,
        mem_limit=None,
    ):
        self.datasets = datasets
        self.sizes = sizes
        self.qualities = qualities
        self.algos = algos
        self.configs = configs
        self.path_to_data_dir = path_to_data_dir
        self.sleep_time = sleep_time
        self.signal_columns = signal_columns
        self.mem_limit = mem_limit or {"Geneformer": 100, "default": 33_000}

        self.experiment_combinations = list(itertools.product(datasets, sizes, qualities, algos))
        random.shuffle(self.experiment_combinations)
        self.experiment_iterator = iter(self.experiment_combinations)
        self.gpu_generators = {}

    def _get_config_for_size(self, algo: str, size: int) -> dict:
        if algo == "Geneformer":
            max_epochs = max(1, int(10 * (10_000_000 / size)))
            print(f"Max epochs for {algo} with {size} cells: {max_epochs}")
            return {"max_epochs": max_epochs, "early_stopping_patience": 3}
        elif algo == "SCVI":
            max_epochs = max(1, int(1 * (10_000_000 / size)))
            print(f"Max epochs for {algo} with {size} cells: {max_epochs}")
            return {"max_epochs": max_epochs, "early_stopping_patience": 3}
        else:
            raise ValueError(f"Algorithm {algo} not supported")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            dataset, size, quality, algo = next(self.experiment_iterator)
        except StopIteration:
            raise StopIteration

        if algo not in self.gpu_generators:
            self.gpu_generators[algo] = self._get_available_gpu(algo)

        device = next(self.gpu_generators[algo])

        if algo == "Geneformer" or algo == "SCVI":
            config = self._get_config_for_size(algo, size)

            job_args = {
                "dataset": dataset,
                "size": size,
                "quality": quality,
                "algo": algo,
                "device": device,
                "max_epochs": config["max_epochs"],
                "early_stopping_patience": config["early_stopping_patience"],
                "signal_columns": self.signal_columns,
            }

            time.sleep(self.sleep_time)
        else:
            job_args = {
                "dataset": dataset,
                "size": size,
                "quality": quality,
                "algo": algo,
                "device": device,
                "signal_columns": self.signal_columns,
            }

        return job_args

    def _get_available_gpu(self, algo: str, timeout: int = 259200):
        """Generator that yields available GPU IDs when they become available."""
        mem_limit = self.mem_limit[algo] if algo in self.mem_limit else self.mem_limit["default"]
        start_time = time.time()

        while True:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
            ).decode()
            available_gpus = []
            for line in gpu_info.strip().split("\n"):
                gpu_id, mem_used = map(int, line.split(", "))
                print(f"GPU {gpu_id} has {mem_used} MB used")
                if mem_used < mem_limit:
                    available_gpus.append(gpu_id)
                    print(f"GPU {available_gpus} is available")

            if available_gpus:
                yield random.choice(available_gpus)
            else:
                elapsed_time = time.time() - start_time
                print(f"No GPUs available, waiting for a free GPU... (elapsed: {elapsed_time/60:.1f} minutes)")
                if elapsed_time > timeout:
                    print(f"Warning: No GPU available after {timeout/60:.1f} minutes, continuing to wait...")
                    start_time = time.time()
                time.sleep(60)


class Experiments:
    def __init__(
        self,
        path_to_data_dir: str,
        datasets: list[str],
        qualities: list[int],
        sizes: list[int],
        algos: list[str],
        signal_columns: list[str],
        device: int = 0,
        seed: int = 42,
    ):
        self.path_to_data_dir: Path = Path(path_to_data_dir)
        self.datasets: list[str] = datasets
        self.qualities: list[int] = qualities
        self.sizes: list[int] = sizes
        self.algos: list[str] = algos
        self.signal_columns: list[str] = signal_columns
        self.device: int = device
        self.seed: int = seed

        self.configs: dict[str, dict[int, dict[str, int]]] = {
            "Geneformer": {
                100: {"max_epochs": 100_000, "early_stopping_patience": 3},
                1_000: {"max_epochs": 10_000, "early_stopping_patience": 3},
                10_000: {"max_epochs": 1_000, "early_stopping_patience": 3},
                100_000: {"max_epochs": 100, "early_stopping_patience": 3},
                1_000_000: {"max_epochs": 10, "early_stopping_patience": 3},
                10_000_000: {"max_epochs": 1, "early_stopping_patience": 3},
            },
            "SCVI": {
                100: {"max_epochs": 100_000, "early_stopping_patience": 3},
                1_000: {"max_epochs": 10_000, "early_stopping_patience": 3},
                10_000: {"max_epochs": 1_000, "early_stopping_patience": 3},
                100_000: {"max_epochs": 100, "early_stopping_patience": 3},
                1_000_000: {"max_epochs": 10, "early_stopping_patience": 3},
                10_000_000: {"max_epochs": 10, "early_stopping_patience": 3},
            },
        }

    def clean_old_results(self):
        for dataset in self.datasets:
            for size in self.sizes:
                for quality in self.qualities:
                    for algo in self.algos:
                        path = self.path_to_data_dir / dataset / f"{size}" / f"{quality}" / "results" / algo
                        print(f"Cleaning old results for {path}")
                        if path.exists():
                            shutil.rmtree(path)

    def make_training_set(
        self,
        remove_old: bool = False,
        download_raw_data: bool = False,
        sample_dataset: bool = False,
        downsample_dataset: bool = False,
    ):

        if remove_old:
            print("Removing old training set")
            for dataset in self.datasets:
                path = self.path_to_data_dir / dataset
                if path.exists():
                    for item in path.iterdir():
                        if item.name in ["raw", "test", "utils"]:
                            continue
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()

        for dataset in self.datasets:

            if download_raw_data:
                print(f"Downloading raw data for {dataset}")
                PrepareData.download_raw_data(dataset_name=dataset, path_to_data_dir=self.path_to_data_dir)

            for size in self.sizes:
                if sample_dataset:
                    PrepareData.sample(
                        sample_size=size,
                        path_to_dataset_dir=self.path_to_data_dir / dataset,
                        output_file=self.path_to_data_dir / dataset / f"{size}" / f"sample.h5ad",
                    )

                for quality in self.qualities:
                    if downsample_dataset:
                        print(f"Downsampling {size} cells for {dataset} with quality {quality}")
                        path_to_noised_sample = self.path_to_data_dir / dataset / f"{size}" / f"{quality}"
                        path_to_noised_sample.mkdir(parents=True, exist_ok=True)

                        try:
                            PrepareData.downsample(
                                ratio=quality,
                                path_to_sample_data=self.path_to_data_dir / dataset / f"{size}" / f"sample.h5ad",
                                output_dir=self.path_to_data_dir / dataset / f"{size}" / f"{quality}",
                            )

                            prepare_data = PrepareData(base_dir=path_to_noised_sample)
                            prepare_data.create_training_set()
                        except Exception as e:
                            print(f"Error downsampling {size} cells for {dataset} with quality {quality}: {e}")
                            continue

    def make_test_set(
        self,
        remove_old: bool = False,
    ):
        """assume train dataset exists, should i remove dows with only zeros?"""
        if remove_old:
            print("Removing old test set")
            for dataset in self.datasets:
                test_dir = self.path_to_data_dir / dataset / "test"
                if test_dir.exists():
                    shutil.rmtree(test_dir)

        for dataset in self.datasets:

            test_dir = self.path_to_data_dir / dataset / "test"
            os.makedirs(test_dir, exist_ok=True)

            unseen_data, selected_indices_pos = self.held_out_sample()
            test_sample_path = test_dir / "sample.h5ad"
            if dataset == "merfish" or dataset == "larry":
                shutil.copy(self.path_to_data_dir / dataset / "raw" / "raw.h5ad", test_sample_path)
                test_adata = ad.read_h5ad(test_sample_path)
                test_adata.uns["test_indices"] = np.array(selected_indices_pos)
                test_adata.write_h5ad(test_sample_path)
            else:
                unseen_data.write_h5ad(test_sample_path)

            signal_dir = self.path_to_data_dir / dataset / "test" / "signals"
            signal_dir.mkdir(parents=True, exist_ok=True)

            for quality in self.qualities:

                downsampled_adata: ad.AnnData = PrepareData.downsample(
                    ratio=quality,
                    path_to_sample_data=test_dir / "sample.h5ad",
                    output_dir=test_dir / f"{float(quality)}" / "preprocessed",
                )
                os.rename(
                    test_dir / f"{quality}" / "preprocessed" / "downsampled.h5ad",
                    test_dir / f"{quality}" / "preprocessed" / "preprocessed.h5ad",
                )

                quality_signal_dir = test_dir / f"{quality}" / "signals"
                quality_signal_dir.mkdir(parents=True, exist_ok=True)

                for signal_column in self.signal_columns:
                    y_path = quality_signal_dir / f"Y_{signal_column}_{quality}.csv"
                    if signal_column == "protein_counts":
                        prot_col = [col for col in downsampled_adata.obs.columns if col.startswith("prot_")]
                        pd.DataFrame(downsampled_adata.obs[prot_col]).to_csv(y_path, index=False)
                    elif signal_column == "cur_idx" or signal_column == "ng_idx":

                        unique_indexes = self.sample_neighbors(downsampled_adata)
                        downsampled_adata.obs["ng_idx"] = unique_indexes.values.astype(str)
                        downsampled_adata.obs["cur_idx"] = downsampled_adata.obs.index.values.astype(str)
                        downsampled_adata.write_h5ad(test_dir / f"{quality}" / "preprocessed" / "preprocessed.h5ad")
                        ng_idx = downsampled_adata.obs["ng_idx"].values.astype(str)
                        cur_idx = downsampled_adata.obs["cur_idx"].values.astype(str)
                        assert set(ng_idx).issubset(set(cur_idx)), "ng_idx has values that are not in cur_idx"
                        pd.DataFrame(dict(cur_idx=cur_idx, ng_idx=ng_idx)).to_csv(y_path, index=False)
                    elif signal_column == "clone" or signal_column == "index" or signal_column == "time":
                        clone = downsampled_adata.obs["clone"].values.astype(str)

                        save_dict = dict(
                            index=downsampled_adata.obs["index"].values.astype(str),
                            clone=clone,
                            time=downsampled_adata.obs["time"].values.astype(int),
                        )
                        pd.DataFrame(save_dict).to_csv(y_path, index=False)
                    else:
                        pd.DataFrame(downsampled_adata.obs[signal_column]).to_csv(y_path, index=False)

    def sample_neighbors(self, data, k_neighbors=5, x="center_x", y="center_y"):
        xy = data.obs[[x, y]]
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(xy)
        _, positions = nbrs.kneighbors(xy)
        nbs = positions[np.arange(len(xy)), np.random.randint(1, high=k_neighbors, size=len(xy))]
        unique_indexes = data.obs.index[nbs]
        assert len(unique_indexes) == len(data.obs.index.unique()), "Some indexes are not unique"
        return unique_indexes

    def held_out_sample(self):
        for dataset in self.datasets:
            indecies = set()
            for size in self.sizes:
                path = self.path_to_data_dir / dataset / f"{size}" / "sample.h5ad"
                data = ad.read_h5ad(path, backed="r")
                indecies.update(data.obs.index)

        raw_data_path = self.path_to_data_dir / dataset / "raw" / "raw.h5ad"
        raw_data = ad.read_h5ad(raw_data_path, backed="r")
        unseen_indices = set(raw_data.obs.index) - indecies
        unseen_indices_list = list(unseen_indices)
        selected_indices = np.random.choice(
            unseen_indices_list, size=min(30_000, len(unseen_indices_list)), replace=False
        )
        print(f"Selected {len(selected_indices)} indices")
        unseen_data = raw_data[selected_indices].to_memory()

        selected_indices_pos = [raw_data.obs.index.get_loc(idx) for idx in selected_indices]
        return unseen_data, selected_indices_pos

    def make_validation_set(
        self,
        remove_old: bool = False,
    ):
        """assume train dataset exists, should i remove dows with only zeros?"""
        if remove_old:
            print("Removing old validation set")
            for dataset in self.datasets:
                validation_dir = self.path_to_data_dir / dataset / "validation"
                if validation_dir.exists():
                    shutil.rmtree(validation_dir)

        for dataset in self.datasets:
            indecies = set()

            for size in tqdm(self.sizes, desc=f"Processing sizes for {dataset}"):
                path = self.path_to_data_dir / dataset / f"{size}" / "sample.h5ad"
                data = ad.read_h5ad(path, backed="r")
                indecies.update(data.obs.index)

            raw_data_path = self.path_to_data_dir / dataset / "raw" / "raw.h5ad"
            raw_data = ad.read_h5ad(raw_data_path, backed="r")
            unseen_indices = set(raw_data.obs.index) - indecies
            unseen_indices_list = list(unseen_indices)

            selected_indices = np.random.choice(
                unseen_indices_list, size=min(1000, len(unseen_indices_list)), replace=False
            )
            print(f"Selected {len(selected_indices)} indices for validation set")

            subset = raw_data[selected_indices].to_memory()
            validation_dir = self.path_to_data_dir / dataset / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            subset.write_h5ad(validation_dir / "sample.h5ad")

            signal_dir = self.path_to_data_dir / dataset / "validation" / "signals"
            signal_dir.mkdir(parents=True, exist_ok=True)

            for quality in self.qualities:
                PrepareData.downsample(
                    ratio=quality,
                    path_to_sample_data=validation_dir / "sample.h5ad",
                    output_dir=validation_dir / f"{float(quality)}" / "preprocessed",
                )
                os.rename(
                    validation_dir / f"{quality}" / "preprocessed" / "downsampled.h5ad",
                    validation_dir / f"{quality}" / "preprocessed" / "preprocessed.h5ad",
                )

    def tokenize_median_files(
        self,
        partition_size,
        recompute_median_files: bool = False,
        train: bool = True,
        validation: bool = True,
        test: bool = True,
        chunk_size: int = 100_000,
        nproc: int = 50,
    ):

        for dataset in self.datasets:

            if recompute_median_files:
                PrepareData(base_dir=self.path_to_data_dir / dataset / f"{partition_size}" / f"1.0").median_files(
                    sizes=self.sizes
                )

            if train:
                for size in self.sizes:

                    for quality in self.qualities:
                        print(f"Tokenizing training set for {dataset} {size} {quality}")
                        prepare_data = PrepareData(base_dir=self.path_to_data_dir / dataset / f"{size}" / f"{quality}")
                        try:
                            start_time = time.time()
                            prepare_data.tokenize(model_input_size=512, chunk_size=chunk_size, nproc=nproc)
                            print(
                                f"Tokenization of {dataset} {size} {quality} took {time.time() - start_time:.2f} seconds"
                            )
                        except Exception as e:
                            print(f"Error tokenizing {dataset} {size} {quality}: {e}")
                            continue
            if validation:
                for quality in self.qualities:

                    prepare_data = PrepareData(base_dir=self.path_to_data_dir / dataset / "validation" / f"{quality}")
                    prepare_data.tokenize(model_input_size=512, chunk_size=chunk_size, nproc=nproc)
            if test:
                for quality in self.qualities:

                    prepare_data = PrepareData(base_dir=self.path_to_data_dir / dataset / "test" / f"{quality}")
                    prepare_data.tokenize(
                        model_input_size=512, signal_columns=self.signal_columns, chunk_size=chunk_size, nproc=nproc
                    )

    def _record_failed_job(self, job_args: dict, error: str, file_path: str = "failed_jobs.txt"):
        """Record failed job configuration to a file with timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file_path, "a") as f:
            f.write(f"\n[{timestamp}] Failed job configuration:\n")
            f.write(f"  Dataset: {job_args.get('dataset')}\n")
            f.write(f"  Size: {job_args.get('size')}\n")
            f.write(f"  Quality: {job_args.get('quality')}\n")
            f.write(f"  Algorithm: {job_args.get('algo')}\n")
            f.write(f"  Device: {job_args.get('device')}\n")
            f.write(f"  Seed: {job_args.get('seed')}\n")
            f.write(f"  Error: {error}\n")
            f.write("-" * 80 + "\n")

    def run(self):
        experiment_iterator = itertools.product(self.datasets, self.sizes, self.qualities, self.algos)

        for dataset, size, quality, algo in experiment_iterator:
            try:
                if algo == "Geneformer" or algo == "SCVI":
                    self.single_job(dataset, size, quality, algo, **self.configs[algo][size], device=self.device)
                else:
                    self.single_job(dataset, size, quality, algo)
            except Exception as e:
                job_args = {
                    "dataset": dataset,
                    "size": size,
                    "quality": quality,
                    "algo": algo,
                    "device": self.device,
                    "seed": self.seed,
                }
                self._record_failed_job(job_args, ".", file_path="failed.txt")
                print(f"Error running {algo} for {dataset} with {size} cells and {quality} quality: {e}")

    def parallel_run(
        self,
        max_workers: int = 40,
        sleep_time: int = 30,
        retrain: bool = True,
        reembed: bool = True,
        recompute_mutual_information: bool = True,
        checkpoint_path: str | None = None,
        mem_limit: dict[str, int] | None = None,
        reembed_checkpoint: str | None = None,
        batch_size_inference: int | None = None,
    ):
        """SCVI needs to run alone since it blocks all of the GPUs"""

        job_iterator = ExperimentJobIterator(
            datasets=self.datasets,
            sizes=self.sizes,
            qualities=self.qualities,
            algos=self.algos,
            configs=self.configs,
            path_to_data_dir=self.path_to_data_dir,
            signal_columns=self.signal_columns,
            sleep_time=sleep_time,
            mem_limit=mem_limit or {"Geneformer": 100, "default": 33_000},
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = {}

            print(f"Submitting initial batch to max workers {max_workers}...")
            jobs_submitted = 0
            for i in range(max_workers):
                try:
                    job_args = next(job_iterator)
                    job_args.update(
                        {
                            "retrain": retrain,
                            "reembed": reembed,
                            "recompute_mutual_information": recompute_mutual_information,
                            "checkpoint_path": checkpoint_path,
                            "reembed_checkpoint": reembed_checkpoint,
                            "batch_size_inference": batch_size_inference,
                            "seed": self.seed,
                        }
                    )
                    future = executor.submit(JobProcessor(**job_args))
                    active_futures[future] = job_args
                    jobs_submitted += 1
                    print(
                        f"Submitted job {jobs_submitted}/{max_workers}: {job_args['algo']} for {job_args['dataset']} ({job_args['size']} cells, quality {job_args['quality']}) on device {job_args['device']}"
                    )
                except StopIteration:
                    print(f"No more jobs available, submitted {jobs_submitted} initial jobs")
                    break

            print(f"Initial batch complete: {len(active_futures)} jobs running concurrently")

            total_jobs = len(list(itertools.product(self.datasets, self.sizes, self.qualities, self.algos)))
            completed_jobs = 0

            while active_futures:
                completed_future = next(as_completed(active_futures))
                completed_job_args = active_futures.pop(completed_future)
                completed_jobs += 1

                try:
                    completed_future.result()
                    print(
                        f"Completed job {completed_jobs}/{total_jobs}: {completed_job_args['algo']} for {completed_job_args['dataset']} ({completed_job_args['size']} cells, quality {completed_job_args['quality']})"
                    )
                except Exception as e:
                    self._record_failed_job(completed_job_args, str(e))
                    print(f"Error in job {completed_job_args}: {e}")

                try:
                    job_args = next(job_iterator)
                    job_args.update(
                        {
                            "retrain": retrain,
                            "reembed": reembed,
                            "recompute_mutual_information": recompute_mutual_information,
                            "checkpoint_path": checkpoint_path,
                            "reembed_checkpoint": reembed_checkpoint,
                            "batch_size_inference": batch_size_inference,
                            "seed": self.seed,
                        }
                    )
                    new_future = executor.submit(JobProcessor(**job_args))
                    active_futures[new_future] = job_args
                    print(
                        f"Submitted new job: {job_args['algo']} for {job_args['dataset']} ({job_args['size']} cells, quality {job_args['quality']}) on device {job_args['device']} | Active jobs: {len(active_futures)}"
                    )
                except StopIteration:
                    print(f"No more jobs to submit. Active jobs: {len(active_futures)}")
                    pass

            print(f"All {completed_jobs} jobs completed!")

    def evaluate_checkpoints_mutual_information(
        self,
        inference_batch_size: int = 100,
        max_epochs: int = 2,
        num_points: int = 20,
        device: int = 0,
    ):

        for dataset_name, size_val, quality_val in itertools.product(self.datasets, self.sizes, self.qualities):
            checkpointed_models = self._get_sorted_and_linspaced_checkpoints_names(
                dataset_name, size_val, quality_val, num_points
            )
            for model_name in checkpointed_models:
                base_dir = self.path_to_data_dir / f"{dataset_name}" / f"{size_val}" / f"{quality_val}"
                print(
                    f"Running Geneformer model {model_name} for {dataset_name} with {size_val} cells and {quality_val} quality on device {device}"
                )
                method = Geneformer(
                    base_dir=base_dir,
                    signal_columns=self.signal_columns,
                    device=device,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=self.seed,
                )
                print(
                    f"Embedding {model_name} for {dataset_name} with {size_val} cells and {quality_val} quality on device {device} and batch size {inference_batch_size}"
                )
                method.embed(inference_batch_size=inference_batch_size)
                print(
                    f"Mutual information for {model_name} for {dataset_name} with {size_val} cells and {quality_val} quality on device {device} and max_epochs {max_epochs}"
                )
                method.mutual_information(max_epochs=max_epochs)

    def _get_sorted_and_linspaced_checkpoints_names(
        self, dataset_name: str, size_val: int, quality_val: float, num_points: int = 20
    ):
        checkpoint_dir = (
            self.path_to_data_dir / dataset_name / str(size_val) / str(quality_val) / "results" / "Geneformer"
        )
        checkpoint_dirs = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint")]
        checkpoint_dirs = sorted(
            checkpoint_dirs,
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name and p.name.split("-")[1].isdigit() else 0,
        )
        if len(checkpoint_dirs) > num_points:
            indices = np.linspace(0, len(checkpoint_dirs) - 1, num_points, dtype=int)
            checkpoint_dirs = [checkpoint_dirs[i] for i in indices]
        checkpointed_models: list[str] = [p.name for p in checkpoint_dirs]
        return checkpointed_models

    def single_job(
        self,
        dataset,
        size,
        quality,
        algo,
        max_epochs: int = None,
        early_stopping_patience: int = None,
        device: int = 0,
        retrain: bool = True,
        reembed: bool = True,
        recompute_mutual_information: bool = True,
        checkpoint_path: str | None = None,
        reembed_checkpoint: str | None = None,
        batch_size_inference: int | None = None,
    ):

        base_dir = self.path_to_data_dir / f"{dataset}" / f"{size}" / f"{quality}"
        print(
            f"Running {algo} for {dataset} with {size} cells and {quality} quality on device {device} (using max_epochs={max_epochs} and early_stopping_patience={early_stopping_patience})"
        )
        if reembed_checkpoint:
            model_name = reembed_checkpoint.split("/")[-1]
        else:
            model_name = None

        if algo == "Geneformer":
            method = Geneformer(
                base_dir=base_dir,
                lengths_path=base_dir / "preprocessed" / "lengths.pkl",
                signal_columns=self.signal_columns,
                device=device,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                dataset_name=self.datasets[0],
                seed=self.seed,
                model_name=model_name,
            )
        elif algo == "RandomProjection":
            method = RandomProjection(
                base_dir=base_dir,
                device=device,
                seed=self.seed,
            )
        elif algo == "SCVI":
            method = SCVI(
                base_dir=base_dir,
                signal_columns=self.signal_columns,
                device=device,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                dataset_name=self.datasets[0],
                seed=self.seed,
            )
        elif algo == "PCA":
            method = PCA(
                base_dir=base_dir,
                signal_columns=self.signal_columns,
                device=device,
                seed=self.seed,
            )

        print(
            f"Running {algo} for {dataset} with {size} cells and {quality} quality on device {device} (using max_epochs={max_epochs} and early_stopping_patience={early_stopping_patience})"
        )
        if retrain:
            method.train()
        if reembed:
            if algo == "Geneformer":
                method.embed(inference_batch_size=batch_size_inference)
            else:
                method.embed()
        if recompute_mutual_information:
            method.mutual_information()

    def evaluate_checkpoints_mutual_information_parallel(
        self,
        max_workers: int = 40,
        sleep_time: int = 30,
        inference_batch_size: int = 100,
        max_epochs: int = 2,
        num_points: int = 20,
        mem_limit: dict[str, int] | None = None,
        remove_old: bool = False,
    ):
        """Evaluate mutual information for checkpoints in parallel using the job processing infrastructure."""

        if remove_old:
            for dataset_name, size_val, quality_val in itertools.product(self.datasets, self.sizes, self.qualities):
                geneformer_path = (
                    self.path_to_data_dir / dataset_name / str(size_val) / str(quality_val) / "results" / "Geneformer"
                )
                if geneformer_path.exists():
                    for checkpoint_dir in geneformer_path.glob("checkpoint-*"):
                        mi_path = checkpoint_dir / "MI"
                        if mi_path.exists():
                            print(mi_path)
                            shutil.rmtree(mi_path)

        all_jobs = []
        for dataset_name, size_val, quality_val in itertools.product(self.datasets, self.sizes, self.qualities):
            checkpointed_models = self._get_sorted_and_linspaced_checkpoints_names(
                dataset_name, size_val, quality_val, num_points
            )
            for model_name in checkpointed_models:
                job = {
                    "dataset": dataset_name,
                    "size": size_val,
                    "quality": quality_val,
                    "algo": "Geneformer",
                    "retrain": False,
                    "reembed": True,
                    "recompute_mutual_information": True,
                    "reembed_checkpoint": str(
                        self.path_to_data_dir
                        / dataset_name
                        / str(size_val)
                        / str(quality_val)
                        / "results"
                        / "Geneformer"
                        / model_name
                    ),
                    "batch_size_inference": inference_batch_size,
                    "max_epochs": max_epochs,
                    "early_stopping_patience": 3,
                    "signal_columns": self.signal_columns,
                    "seed": self.seed,
                }
                all_jobs.append(job)

        random.shuffle(all_jobs)

        class CheckpointJobIterator:
            def __init__(self, jobs, mem_limit):
                self.jobs = jobs
                self.job_index = 0
                self.gpu_generators = {}
                self.mem_limit = mem_limit or {"Geneformer": 100, "default": 33_000}

            def __iter__(self):
                return self

            def __next__(self):
                if self.job_index >= len(self.jobs):
                    raise StopIteration

                job = self.jobs[self.job_index]
                self.job_index += 1

                if "Geneformer" not in self.gpu_generators:
                    self.gpu_generators["Geneformer"] = self._get_available_gpu("Geneformer")

                job["device"] = next(self.gpu_generators["Geneformer"])
                return job

            def _get_available_gpu(self, algo: str, timeout: int = 259200):
                mem_limit = self.mem_limit[algo] if algo in self.mem_limit else self.mem_limit["default"]
                start_time = time.time()

                while True:
                    gpu_info = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
                    ).decode()
                    available_gpus = []
                    for line in gpu_info.strip().split("\n"):
                        gpu_id, mem_used = map(int, line.split(", "))
                        if mem_used < mem_limit:
                            available_gpus.append(gpu_id)

                    if available_gpus:
                        yield random.choice(available_gpus)
                    else:
                        elapsed_time = time.time() - start_time
                        print(f"No GPUs available, waiting for a free GPU... (elapsed: {elapsed_time/60:.1f} minutes)")
                        if elapsed_time > timeout:
                            print(f"Warning: No GPU available after {timeout/60:.1f} minutes, continuing to wait...")
                            start_time = time.time()
                        time.sleep(60)

        job_iterator = CheckpointJobIterator(all_jobs, mem_limit)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = {}

            print(f"Submitting initial batch to max workers {max_workers}...")
            jobs_submitted = 0
            for i in range(max_workers):
                try:
                    job_args = next(job_iterator)
                    future = executor.submit(JobProcessor(**job_args))
                    active_futures[future] = job_args
                    jobs_submitted += 1
                    print(
                        f"Submitted checkpoint evaluation job {jobs_submitted}/{max_workers}: Model {job_args['reembed_checkpoint']} for {job_args['dataset']} on device {job_args['device']}"
                    )
                except StopIteration:
                    print(f"No more jobs available, submitted {jobs_submitted} initial jobs")
                    break

            print(f"Initial batch complete: {len(active_futures)} jobs running concurrently")

            total_jobs = len(all_jobs)
            completed_jobs = 0

            while active_futures:
                completed_future = next(as_completed(active_futures))
                completed_job_args = active_futures.pop(completed_future)
                completed_jobs += 1

                try:
                    completed_future.result()
                    print(
                        f"Completed checkpoint evaluation job {completed_jobs}/{total_jobs}: Model {completed_job_args['reembed_checkpoint']}"
                    )
                except Exception as e:
                    self._record_failed_job(completed_job_args, str(e))
                    print(f"Error in checkpoint evaluation job {completed_job_args['reembed_checkpoint']}: {e}")

                try:
                    job_args = next(job_iterator)
                    new_future = executor.submit(JobProcessor(**job_args))
                    active_futures[new_future] = job_args
                    print(
                        f"Submitted new checkpoint evaluation job: Model {job_args['reembed_checkpoint']} on device {job_args['device']} | Active jobs: {len(active_futures)}"
                    )
                except StopIteration:
                    print(f"No more jobs to submit. Active jobs: {len(active_futures)}")
                    pass

            print(f"All {completed_jobs} checkpoint evaluation jobs completed!")


class JobProcessor:
    def __init__(self, **kwargs):
        dataset = kwargs["dataset"]
        size = kwargs["size"]
        quality = kwargs["quality"]
        algo = kwargs["algo"]
        max_epochs = kwargs.get("max_epochs", 1)
        early_stopping_patience = kwargs.get("early_stopping_patience", 1)
        device = kwargs.get("device", 0)
        signal_columns = kwargs.get("signal_columns", [])
        retrain = kwargs.get("retrain", True)
        reembed = kwargs.get("reembed", True)
        recompute_mutual_information = kwargs.get("recompute_mutual_information", True)
        checkpoint_path = kwargs.get("checkpoint_path", None)
        reembed_checkpoint = kwargs.get("reembed_checkpoint", None)
        batch_size_inference = kwargs.get("batch_size_inference", None)
        seed = kwargs.get("seed", 42)

        print(dataset, size, quality, algo, max_epochs, early_stopping_patience, device)
        path_to_data_dir = Path("/home/jupyter/igor_repos/noise_scaling_laws/data")

        self.cmd = [
            "python",
            "/home/jupyter/igor_repos/noise_scaling_laws/single_job.py",
            "--sizes",
            str(size),
            "--qualities",
            str(quality),
            "--algos",
            algo,
            "--base_dir",
            str(path_to_data_dir),
            "--device",
            str(device),
            "--max_epochs",
            str(max_epochs),
            "--early_stopping_patience",
            str(early_stopping_patience),
            "--dataset",
            dataset,
            "--retrain",
            str(retrain).lower(),
            "--reembed",
            str(reembed).lower(),
            "--recompute_mutual_information",
            str(recompute_mutual_information).lower(),
            "--seed",
            str(seed),
        ]

        if signal_columns:
            self.cmd.extend(["--signal_columns"] + signal_columns)

        if checkpoint_path:
            self.cmd.extend(["--checkpoint_path", str(checkpoint_path)])

        if reembed_checkpoint:
            self.cmd.extend(["--reembed_checkpoint", str(reembed_checkpoint)])

        if batch_size_inference:
            self.cmd.extend(["--batch_size_inference", str(batch_size_inference)])

    def __call__(self):
        try:
            subprocess.run(self.cmd, check=True)
        except subprocess.CalledProcessError as e:
            cmd_args = {}
            for i, arg in enumerate(self.cmd):
                if arg == "--dataset":
                    cmd_args["dataset"] = self.cmd[i + 1]
                elif arg == "--sizes":
                    cmd_args["size"] = int(self.cmd[i + 1])
                elif arg == "--qualities":
                    cmd_args["quality"] = float(self.cmd[i + 1])
                elif arg == "--algos":
                    cmd_args["algo"] = self.cmd[i + 1]
                elif arg == "--device":
                    cmd_args["device"] = int(self.cmd[i + 1])
                elif arg == "--seed":
                    cmd_args["seed"] = int(self.cmd[i + 1])

            experiments = Experiments(
                path_to_data_dir="/home/jupyter/igor_repos/noise_scaling_laws/data",
                datasets=[cmd_args["dataset"]],
                qualities=[cmd_args["quality"]],
                sizes=[cmd_args["size"]],
                algos=[cmd_args["algo"]],
                signal_columns=[],
                device=cmd_args["device"],
                seed=cmd_args["seed"],
            )
            experiments._record_failed_job(cmd_args, str(e))
            raise e
