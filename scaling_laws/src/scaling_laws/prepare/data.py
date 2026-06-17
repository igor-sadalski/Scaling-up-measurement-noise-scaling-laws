from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import os
import pickle
import shutil
import time
import traceback
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

    def prepare_for_state(self, profile_name: str, split: str = "train") -> Path:
        """Create state_data/ directory and write a CSV manifest for this split.

        Args:
            profile_name: The profile name used by ``state emb preprocess``.
            split: One of 'train', 'val', or 'test'.

        Returns:
            Path to the written CSV manifest.
        """
        state_data_dir = self.preprocessed / "state_data"
        state_data_dir.mkdir(parents=True, exist_ok=True)

        h5ad_path = self.preprocessed / "preprocessed.h5ad"
        csv_path = state_data_dir / f"{split}.csv"
        csv_path.write_text(
            f"species,path,names\nhuman,{h5ad_path},{profile_name}_{split}\n"
        )
        print(f"  State {split} manifest: {csv_path}")
        return csv_path

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
            import urllib.request

            shendure_url = "https://datasets.cellxgene.cziscience.com/a5a85963-8004-41a1-8eb5-ca65266d89c3.h5ad"
            shendure_file = raw_data_path / "raw.h5ad"

            if not shendure_file.exists():
                print(f"Downloading shendure data from {shendure_url}")
                print(f"Saving to {shendure_file}")
                urllib.request.urlretrieve(shendure_url, shendure_file)
                print("Download complete")
            else:
                print(f"Using existing shendure data at {shendure_file}")

            adata = ad.read_h5ad(str(shendure_file), backed=True)

        elif dataset_name == "merfish":
            raw_dir = path_to_data_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            # Raw merfish CSVs live next to the dataset's `raw/` dir; the older
            # absolute paths under /home/igor/exploration/ are dead.
            meta_csv = raw_dir / "S1R1_meta.csv"
            cxg_csv = raw_dir / "S1R1_cxg.csv"
            meta = pd.read_csv(meta_csv, index_col=0)
            cxg = pd.read_csv(cxg_csv, index_col=0)
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


def _prepare_state_data_single(
    path_to_data_dir, dataset, size, quality,
    esm_embeddings_path, state_python, state_package_dir, state_defaults_yaml,
):
    """Prepare STATE preprocessing profile for a single (dataset, size, quality) combo.

    Top-level function so it can be pickled by ProcessPoolExecutor.
    """
    print(f"\n=== Preparing State data for {dataset} / {size} / {quality} ===")

    profile_name = f"scaling_{dataset}_{size}_{quality}".replace(".", "_")

    train_base = path_to_data_dir / dataset / f"{size}" / f"{quality}"
    val_base = path_to_data_dir / dataset / "validation" / f"{quality}"
    test_base = path_to_data_dir / dataset / "test" / f"{quality}"

    pd_train = PrepareData(base_dir=str(train_base))
    pd_val = PrepareData(base_dir=str(val_base))
    pd_test = PrepareData(base_dir=str(test_base))

    train_csv = pd_train.prepare_for_state(profile_name, split="train")
    pd_val.prepare_for_state(profile_name, split="val")
    pd_test.prepare_for_state(profile_name, split="test")

    profile_dir = pd_train.preprocessed / "state_data"
    combined_val_csv = profile_dir / "val_combined.csv"
    val_h5ad = pd_val.preprocessed / "preprocessed.h5ad"
    test_h5ad = pd_test.preprocessed / "preprocessed.h5ad"
    combined_val_csv.write_text(
        f"species,path,names\n"
        f"human,{val_h5ad},{profile_name}_val\n"
        f"human,{test_h5ad},{profile_name}_test\n"
    )

    val_only_csv = profile_dir / "val_only.csv"
    val_only_csv.write_text(
        f"species,path,names\n"
        f"human,{val_h5ad},{profile_name}_val\n"
    )

    config_path = profile_dir / "state_config.yaml"
    shutil.copy(state_defaults_yaml, config_path)

    cmd = [
        str(state_python), "-m", "state", "emb", "preprocess",
        "--profile-name", profile_name,
        "--train-csv", str(train_csv),
        "--val-csv", str(combined_val_csv),
        "--output-dir", str(profile_dir),
        "--config-file", str(config_path),
    ]
    if esm_embeddings_path:
        cmd.extend(["--all-embeddings", esm_embeddings_path])
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(state_package_dir), check=True)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    preprocessed_val_csv = Path(cfg.dataset[profile_name].val)
    val_only_out = profile_dir / f"val_only_{profile_name}.csv"
    df_val = pd.read_csv(str(preprocessed_val_csv))
    df_val = df_val[df_val["names"].str.endswith("_val")]
    df_val.to_csv(str(val_only_out), index=False)
    assert val_only_out.exists(), f"val_only CSV not written: {val_only_out}"
    cfg.dataset[profile_name].val = str(val_only_out)
    cfg.dataset[profile_name].num_datasets = len(
        pd.read_csv(str(cfg.dataset[profile_name].train))
    ) + len(df_val)
    OmegaConf.save(cfg, str(config_path))
    cfg_verify = OmegaConf.load(str(config_path))
    assert str(val_only_out) in cfg_verify.dataset[profile_name].val, (
        f"Config patching failed: val still points to {cfg_verify.dataset[profile_name].val}"
    )
    print(f"  Config patched: val uses val-only (no test leakage)")
    print(f"  Profile saved to {profile_dir}")


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

    def _get_config_for_size(self, algo: str, size: int, dataset: str | None = None) -> dict:
        max_size = max(self.sizes)
        if algo == "Geneformer":
            max_epochs = max(1, int(10 * (max_size / size)))
            print(f"Max epochs for {algo} with {size} cells (max_size={max_size}): {max_epochs}")
            return {"max_epochs": max_epochs, "early_stopping_patience": 3}
        elif algo == "SCVI":
            max_epochs = max(1, int(1 * (max_size / size)))
            print(f"Max epochs for {algo} with {size} cells (max_size={max_size}): {max_epochs}")
            return {"max_epochs": max_epochs, "early_stopping_patience": 3}
        elif algo == "State":
            max_steps = 15_000
            print(f"Max steps for {algo} with {size} cells: {max_steps}; early stopping patience=5")
            return {"max_steps": max_steps, "early_stopping_patience": 5}
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

        if algo in ("Geneformer", "SCVI", "State"):
            config = self._get_config_for_size(algo, size, dataset=dataset)

            job_args = {
                "dataset": dataset,
                "size": size,
                "quality": quality,
                "algo": algo,
                "device": device,
                "early_stopping_patience": config["early_stopping_patience"],
                "signal_columns": self.signal_columns,
                "path_to_data_dir": self.path_to_data_dir,
            }
            if "max_epochs" in config:
                job_args["max_epochs"] = config["max_epochs"]
            if "max_steps" in config:
                job_args["max_steps"] = config["max_steps"]

            time.sleep(self.sleep_time)
        else:
            job_args = {
                "dataset": dataset,
                "size": size,
                "quality": quality,
                "algo": algo,
                "device": device,
                "signal_columns": self.signal_columns,
                "path_to_data_dir": self.path_to_data_dir,
            }

        return job_args

    def _get_available_gpu(self, algo: str, timeout: int = 259200):
        """Generator that yields available GPU IDs when they become available."""
        mem_limit = self.mem_limit.get(algo) or self.mem_limit.get("default") or 0

        # When mem_limit is disabled (<=0), just round-robin GPUs without
        # memory checks.  This avoids blocking when jobs_per_gpu handles
        # concurrency externally.
        if mem_limit <= 0:
            gpu_idx = 0
            while True:
                yield gpu_idx % 8  # placeholder; device is overridden by parallel_run
                gpu_idx += 1

        start_time = time.time()

        def _get_gpu_memory_info():
            """Get GPU memory info, supporting both NVIDIA and ROCm."""
            try:
                gpu_info = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL
                ).decode()
                gpu_data = []
                for line in gpu_info.strip().split("\n"):
                    if line.strip():
                        gpu_id, mem_used = map(int, line.split(", "))
                        gpu_data.append((gpu_id, mem_used))
                return gpu_data
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    gpu_info = subprocess.check_output(
                        ["rocm-smi", "--alldevices", "--showmemuse", "--csv"],
                        stderr=subprocess.PIPE
                    ).decode()
                    gpu_data = []
                    lines = gpu_info.strip().split("\n")
                    
                    if len(lines) < 2:
                        print(f"Warning: rocm-smi returned insufficient data. Output: {gpu_info[:200]}")
                        return []
                    
                    header = lines[0].lower()
                    mem_col_idx = None
                    gpu_id_col_idx = 0
                    
                    for i, col in enumerate(header.split(",")):
                        col = col.strip()
                        if "vram" in col or ("memory" in col and "allocated" in col):
                            mem_col_idx = i
                            break
                    
                    if mem_col_idx is None:
                        mem_col_idx = 1
                    
                    for i, line in enumerate(lines[1:], 1):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) > max(gpu_id_col_idx, mem_col_idx):
                            try:
                                device_str = parts[gpu_id_col_idx]
                                if device_str.startswith("card"):
                                    gpu_id = int(device_str.replace("card", ""))
                                else:
                                    gpu_id = int(device_str)
                                
                                mem_str = parts[mem_col_idx].strip()
                                mem_used = int(float(mem_str))
                                
                                gpu_data.append((gpu_id, mem_used))
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Failed to parse GPU info from line {i}: {line[:100]}, error: {e}")
                                continue
                    
                    if not gpu_data:
                        print(f"Warning: No GPU data parsed from rocm-smi. Output: {gpu_info[:500]}")
                    
                    return gpu_data
                except subprocess.CalledProcessError as e:
                    print(f"Error running rocm-smi: {e.stderr.decode() if e.stderr else str(e)}")
                    return []
                except FileNotFoundError:
                    print("Warning: rocm-smi command not found")
                    return []
                except Exception as e:
                    print(f"Unexpected error getting GPU info: {e}")
                    return []

        while True:
            gpu_data = _get_gpu_memory_info()
            
            if not gpu_data:
                print("Warning: Could not detect any GPUs. Trying fallback method...")
                try:
                    result = subprocess.run(
                        ["rocm-smi"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        gpu_ids = []
                        for line in result.stdout.split("\n"):
                            if "GPU[" in line:
                                try:
                                    start = line.find("GPU[") + 4
                                    end = line.find("]", start)
                                    if end > start:
                                        gpu_id = int(line[start:end])
                                        if 0 <= gpu_id < 16:
                                            if gpu_id not in gpu_ids:
                                                gpu_ids.append(gpu_id)
                                                gpu_data.append((gpu_id, 0))
                                except (ValueError, IndexError):
                                    continue
                        if gpu_data:
                            print(f"Fallback: Detected {len(gpu_data)} GPUs (IDs: {sorted(gpu_ids)}), assuming 0 MB used")
                        else:
                            print(f"Fallback: Could not parse GPU IDs from rocm-smi output")
                except Exception as e:
                    print(f"Fallback method also failed: {e}")
            
            available_gpus = []
            for gpu_id, mem_used in gpu_data:
                print(f"GPU {gpu_id} has {mem_used} MB used (limit: {mem_limit} MB)")
                if mem_used < mem_limit:
                    available_gpus.append(gpu_id)
            
            if available_gpus:
                selected_gpu = random.choice(available_gpus)
                print(f"Selected GPU {selected_gpu} from available GPUs: {available_gpus}")
                yield selected_gpu
            else:
                if gpu_data:
                    print(f"No GPUs available (all {len(gpu_data)} GPUs exceed memory limit of {mem_limit} MB)")
                else:
                    print("No GPUs detected at all")
                elapsed_time = time.time() - start_time
                print(f"Waiting for a free GPU... (elapsed: {elapsed_time/60:.1f} minutes)")
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
            "State": {
                # Fixed step budget (no early stopping); steps ≈ max_epochs × (size // 64)
                100: {"max_epochs": 100_000, "early_stopping_patience": 0, "max_steps": 100_000},
                1_000: {"max_epochs": 10_000, "early_stopping_patience": 0, "max_steps": 150_000},
                10_000: {"max_epochs": 1_000, "early_stopping_patience": 0, "max_steps": 156_000},
                100_000: {"max_epochs": 100, "early_stopping_patience": 0, "max_steps": 156_200},
                1_000_000: {"max_epochs": 10, "early_stopping_patience": 0, "max_steps": 156_250},
                10_000_000: {"max_epochs": 10, "early_stopping_patience": 0, "max_steps": 1_562_500},
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

    def prepare_state_data(self, esm_embeddings_path: str | None = None, max_workers: int = 1):
        """Run ``state emb preprocess`` for every dataset / size / quality.

        Parameters
        ----------
        esm_embeddings_path : str, optional
            Path to a ``.pt`` file mapping gene names to ESM embedding
            tensors.  When provided, STATE will use these embeddings
            instead of one-hot vectors.  Defaults to the project-wide
            merged ESM file at ``data/other/esm/merged_esm_embeddings.pt``.
        max_workers : int, optional
            Number of parallel workers for preprocessing.  Each
            (dataset, size, quality) combo writes to its own directory
            so they can safely run concurrently.  Default is 1
            (sequential).

        For each (dataset, size, quality) combination this method:
        1. Writes CSV manifests for train, validation, and test splits
           into ``{size}/{quality}/preprocessed/state_data/``.
        2. Copies the default STATE config.
        3. Calls ``state emb preprocess`` which creates one-hot gene
           embeddings, per-dataset mappings, and valid-gene masks — all
           saved into the train split's ``state_data/`` directory.

        After this, ``State.train()`` can skip preprocessing and go
        straight to ``state emb fit``.
        """
        from scaling_laws.paths import STATE_PYTHON, STATE_PACKAGE_DIR, STATE_DEFAULTS_YAML
        state_python = STATE_PYTHON
        state_package_dir = STATE_PACKAGE_DIR
        state_defaults_yaml = STATE_DEFAULTS_YAML

        # Resolve ESM embeddings path
        if esm_embeddings_path is None:
            default_esm = self.path_to_data_dir / "other" / "esm" / "merged_esm_embeddings.pt"
            if default_esm.exists():
                esm_embeddings_path = str(default_esm)
                print(f"  Using ESM embeddings: {esm_embeddings_path}")
            else:
                print("  No ESM embeddings found, falling back to one-hot")

        combos = [
            (dataset, size, quality)
            for dataset in self.datasets
            for size in self.sizes
            for quality in self.qualities
        ]

        if max_workers <= 1:
            for dataset, size, quality in combos:
                _prepare_state_data_single(
                    self.path_to_data_dir, dataset, size, quality,
                    esm_embeddings_path, state_python, state_package_dir, state_defaults_yaml,
                )
        else:
            print(f"  Preprocessing {len(combos)} combos with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _prepare_state_data_single,
                        self.path_to_data_dir, dataset, size, quality,
                        esm_embeddings_path, state_python, state_package_dir, state_defaults_yaml,
                    ): (dataset, size, quality)
                    for dataset, size, quality in combos
                }
                for future in as_completed(futures):
                    dataset, size, quality = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  FAILED: {dataset} / {size} / {quality}: {e}")

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
                if algo in ("Geneformer", "SCVI", "State"):
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

    @staticmethod
    def _detect_gpus() -> list[int]:
        """Return list of GPU IDs visible on this machine."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode()
            return [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
        except Exception:
            return list(range(8))

    def parallel_run(
        self,
        max_workers: int = 40,
        sleep_time: int = 30,
        retrain: bool = True,
        reembed: bool = True,
        recompute_mutual_information: bool = True,
        recompute_loss: bool = False,
        checkpoint_path: str | None = None,
        mem_limit: dict[str, int] | None = None,
        reembed_checkpoint: str | None = None,
        batch_size_inference: int | None = None,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        max_steps: int | None = None,
        jobs_per_gpu: int = 0,
        log_dir: str | None = None,
    ):
        """Run experiments in parallel across GPUs.

        Parameters
        ----------
        jobs_per_gpu : int
            0 — disabled (default): GPU allocation is handled by the
            iterator's memory-based heuristic via ``mem_limit``.
            1 — one job per GPU at a time (exclusive).
            2, 3, … — up to N concurrent jobs per GPU.
            The number of GPUs is auto-detected and ``max_workers``
            is capped to ``jobs_per_gpu * num_gpus``.
        log_dir : str, optional
            Directory to save per-job log files.  Each job writes its
            stdout/stderr to ``<log_dir>/<algo>_<size>_<quality>.txt``.
            If None, no logs are saved.
        """
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving job logs to: {log_dir}")

        # When jobs_per_gpu > 0, GPU assignment is handled by the slot pool
        # so disable the iterator's memory-based GPU blocking to avoid stalls.
        if jobs_per_gpu > 0:
            effective_mem_limit = {"default": 0}
        else:
            effective_mem_limit = mem_limit or {"Geneformer": 100, "default": 33_000}

        job_iterator = ExperimentJobIterator(
            datasets=self.datasets,
            sizes=self.sizes,
            qualities=self.qualities,
            algos=self.algos,
            configs=self.configs,
            path_to_data_dir=self.path_to_data_dir,
            signal_columns=self.signal_columns,
            sleep_time=sleep_time,
            mem_limit=effective_mem_limit,
        )

        if jobs_per_gpu > 0:
            all_gpus = self._detect_gpus()
            # Build a slot pool: each GPU appears jobs_per_gpu times
            gpu_slots = all_gpus * jobs_per_gpu
            max_workers = len(gpu_slots)
            free_gpus = list(gpu_slots)
            print(f"GPU scheduling: {len(all_gpus)} GPUs x {jobs_per_gpu} jobs/GPU = max_workers={max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = {}

            print(f"Submitting initial batch to max workers {max_workers}...")
            single_job_path = Path(__file__).parent.parent.parent.parent.parent / "single_job.py"
            jobs_submitted = 0
            for i in range(max_workers):
                try:
                    job_args = next(job_iterator)
                    job_args.update(
                        {
                            "retrain": retrain,
                            "reembed": reembed,
                            "recompute_mutual_information": recompute_mutual_information,
                            "recompute_loss": recompute_loss,
                            "checkpoint_path": checkpoint_path,
                            "reembed_checkpoint": reembed_checkpoint,
                            "batch_size_inference": batch_size_inference,
                            "seed": self.seed,
                            "single_job_path": single_job_path,
                            "log_dir": str(log_dir) if log_dir is not None else None,
                        }
                    )
                    if max_epochs is not None:
                        job_args["max_epochs"] = max_epochs
                    if early_stopping_patience is not None:
                        job_args["early_stopping_patience"] = early_stopping_patience
                    if max_steps is not None:
                        job_args["max_steps"] = max_steps
                    if jobs_per_gpu > 0:
                        job_args["device"] = free_gpus.pop(0)
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
            failed_jobs = 0
            pbar = tqdm(total=total_jobs, desc="Jobs", unit="job")

            while active_futures:
                completed_future = next(as_completed(active_futures))
                completed_job_args = active_futures.pop(completed_future)
                completed_jobs += 1
                pbar.update(1)

                if jobs_per_gpu > 0:
                    free_gpus.append(completed_job_args["device"])

                try:
                    completed_future.result()
                    pbar.set_postfix(done=completed_jobs, failed=failed_jobs, active=len(active_futures))
                except Exception as e:
                    failed_jobs += 1
                    pbar.set_postfix(done=completed_jobs, failed=failed_jobs, active=len(active_futures))
                    self._record_failed_job(completed_job_args, str(e))
                    tqdm.write(f"FAIL: {completed_job_args['algo']} {completed_job_args['size']}x{completed_job_args['quality']}: {e}")

                try:
                    job_args = next(job_iterator)
                    job_args.update(
                        {
                            "retrain": retrain,
                            "reembed": reembed,
                            "recompute_mutual_information": recompute_mutual_information,
                            "recompute_loss": recompute_loss,
                            "checkpoint_path": checkpoint_path,
                            "reembed_checkpoint": reembed_checkpoint,
                            "batch_size_inference": batch_size_inference,
                            "seed": self.seed,
                            "single_job_path": single_job_path,
                            "log_dir": str(log_dir) if log_dir is not None else None,
                        }
                    )
                    if max_epochs is not None:
                        job_args["max_epochs"] = max_epochs
                    if early_stopping_patience is not None:
                        job_args["early_stopping_patience"] = early_stopping_patience
                    if max_steps is not None:
                        job_args["max_steps"] = max_steps
                    if jobs_per_gpu > 0:
                        job_args["device"] = free_gpus.pop(0)
                    new_future = executor.submit(JobProcessor(**job_args))
                    active_futures[new_future] = job_args
                    print(
                        f"Submitted new job: {job_args['algo']} for {job_args['dataset']} ({job_args['size']} cells, quality {job_args['quality']}) on device {job_args['device']} | Active jobs: {len(active_futures)}"
                    )
                except StopIteration:
                    print(f"No more jobs to submit. Active jobs: {len(active_futures)}")
                    pass

            pbar.close()
            print(f"All {completed_jobs} jobs completed! ({failed_jobs} failed)")

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
        max_steps: int | None = None,
        device: int = 0,
        retrain: bool = True,
        reembed: bool = True,
        recompute_mutual_information: bool = True,
        checkpoint_path: str | None = None,
        reembed_checkpoint: str | None = None,
        batch_size_inference: int | None = None,
        recompute_loss: bool = False,
    ):

        base_dir = self.path_to_data_dir / f"{dataset}" / f"{size}" / f"{quality}"
        print(
            f"Running {algo} for {dataset} with {size} cells and {quality} quality on device {device} (using max_epochs={max_epochs} and early_stopping_patience={early_stopping_patience})"
        )
        if reembed_checkpoint:
            model_name = reembed_checkpoint.split("/")[-1]
        else:
            model_name = "model"

        if algo == "Geneformer":
            gf_kwargs = dict(
                base_dir=base_dir,
                lengths_path=base_dir / "preprocessed" / "lengths.pkl",
                signal_columns=self.signal_columns,
                device=device,
                dataset_name=self.datasets[0],
                seed=self.seed,
                model_name=model_name,
            )
            if max_epochs is not None:
                gf_kwargs["max_epochs"] = max_epochs
            if early_stopping_patience is not None:
                gf_kwargs["early_stopping_patience"] = early_stopping_patience
            method = Geneformer(**gf_kwargs)
        elif algo == "RandomProjection":
            method = RandomProjection(
                base_dir=base_dir,
                device=device,
                seed=self.seed,
            )
        elif algo == "SCVI":
            scvi_kwargs = dict(
                base_dir=base_dir,
                signal_columns=self.signal_columns,
                device=device,
                dataset_name=self.datasets[0],
                seed=self.seed,
            )
            if max_epochs is not None:
                scvi_kwargs["max_epochs"] = max_epochs
            if early_stopping_patience is not None:
                scvi_kwargs["early_stopping_patience"] = early_stopping_patience
            method = SCVI(**scvi_kwargs)
        elif algo == "PCA":
            method = PCA(
                base_dir=base_dir,
                signal_columns=self.signal_columns,
                device=device,
                seed=self.seed,
            )
        elif algo == "State":
            state_kwargs = dict(
                base_dir=base_dir,
                device=device,
                dataset_name=self.datasets[0],
                seed=self.seed,
            )
            if max_steps is not None:
                state_kwargs["max_steps"] = max_steps
            elif max_epochs is not None:
                state_kwargs["max_epochs"] = max_epochs
            if early_stopping_patience is not None:
                state_kwargs["early_stopping_patience"] = early_stopping_patience
            method = State(**state_kwargs)

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
        if recompute_loss:
            method.compute_test_loss()

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
                    "path_to_data_dir": self.path_to_data_dir,
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

                def _get_gpu_memory_info():
                    """Get GPU memory info, supporting both NVIDIA and ROCm."""
                    try:
                        gpu_info = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                            stderr=subprocess.DEVNULL
                        ).decode()
                        gpu_data = []
                        for line in gpu_info.strip().split("\n"):
                            if line.strip():
                                gpu_id, mem_used = map(int, line.split(", "))
                                gpu_data.append((gpu_id, mem_used))
                        return gpu_data
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        try:
                            gpu_info = subprocess.check_output(
                                ["rocm-smi", "--alldevices", "--showmemuse", "--csv"],
                                stderr=subprocess.PIPE
                            ).decode()
                            gpu_data = []
                            lines = gpu_info.strip().split("\n")
                            
                            if len(lines) < 2:
                                print(f"Warning: rocm-smi returned insufficient data. Output: {gpu_info[:200]}")
                                return []
                            
                            header = lines[0].lower()
                            mem_col_idx = None
                            gpu_id_col_idx = 0
                            
                            for i, col in enumerate(header.split(",")):
                                col = col.strip()
                                if "vram" in col or ("memory" in col and "allocated" in col):
                                    mem_col_idx = i
                                    break
                            
                            if mem_col_idx is None:
                                mem_col_idx = 1
                            
                            for i, line in enumerate(lines[1:], 1):
                                if not line.strip():
                                    continue
                                parts = [p.strip() for p in line.split(",")]
                                if len(parts) > max(gpu_id_col_idx, mem_col_idx):
                                    try:
                                        device_str = parts[gpu_id_col_idx]
                                        if device_str.startswith("card"):
                                            gpu_id = int(device_str.replace("card", ""))
                                        else:
                                            gpu_id = int(device_str)
                                        
                                        mem_str = parts[mem_col_idx].strip()
                                        mem_used = int(float(mem_str))
                                        
                                        gpu_data.append((gpu_id, mem_used))
                                    except (ValueError, IndexError) as e:
                                        print(f"Warning: Failed to parse GPU info from line {i}: {line[:100]}, error: {e}")
                                        continue
                            
                            if not gpu_data:
                                print(f"Warning: No GPU data parsed from rocm-smi. Output: {gpu_info[:500]}")
                            
                            return gpu_data
                        except subprocess.CalledProcessError as e:
                            print(f"Error running rocm-smi: {e.stderr.decode() if e.stderr else str(e)}")
                            return []
                        except FileNotFoundError:
                            print("Warning: rocm-smi command not found")
                            return []
                        except Exception as e:
                            print(f"Unexpected error getting GPU info: {e}")
                            return []

                while True:
                    gpu_data = _get_gpu_memory_info()
                    
                    if not gpu_data:
                        print("Warning: Could not detect any GPUs. Trying fallback method...")
                        try:
                            result = subprocess.run(
                                ["rocm-smi"],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0:
                                gpu_ids = []
                                for line in result.stdout.split("\n"):
                                    if "GPU[" in line:
                                        try:
                                            start = line.find("GPU[") + 4
                                            end = line.find("]", start)
                                            if end > start:
                                                gpu_id = int(line[start:end])
                                                if 0 <= gpu_id < 16:
                                                    if gpu_id not in gpu_ids:
                                                        gpu_ids.append(gpu_id)
                                                        gpu_data.append((gpu_id, 0))
                                        except (ValueError, IndexError):
                                            continue
                                if gpu_data:
                                    print(f"Fallback: Detected {len(gpu_data)} GPUs (IDs: {sorted(gpu_ids)}), assuming 0 MB used")
                                else:
                                    print(f"Fallback: Could not parse GPU IDs from rocm-smi output")
                        except Exception as e:
                            print(f"Fallback method also failed: {e}")
                    
                    available_gpus = []
                    for gpu_id, mem_used in gpu_data:
                        if mem_used < mem_limit:
                            available_gpus.append(gpu_id)

                    if available_gpus:
                        selected_gpu = random.choice(available_gpus)
                        yield selected_gpu
                    else:
                        if gpu_data:
                            print(f"No GPUs available (all {len(gpu_data)} GPUs exceed memory limit of {mem_limit} MB)")
                        else:
                            print("No GPUs detected at all")
                        elapsed_time = time.time() - start_time
                        print(f"Waiting for a free GPU... (elapsed: {elapsed_time/60:.1f} minutes)")
                        if elapsed_time > timeout:
                            print(f"Warning: No GPU available after {timeout/60:.1f} minutes, continuing to wait...")
                            start_time = time.time()
                        time.sleep(60)

        job_iterator = CheckpointJobIterator(all_jobs, mem_limit)
        single_job_path = Path(__file__).parent.parent.parent.parent.parent / "single_job.py"

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = {}

            print(f"Submitting initial batch to max workers {max_workers}...")
            jobs_submitted = 0
            for i in range(max_workers):
                try:
                    job_args = next(job_iterator)
                    job_args["single_job_path"] = single_job_path
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
                    job_args["single_job_path"] = single_job_path
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
        max_steps = kwargs.get("max_steps", None)
        device = kwargs.get("device", 0)
        signal_columns = kwargs.get("signal_columns", [])
        retrain = kwargs.get("retrain", True)
        reembed = kwargs.get("reembed", True)
        recompute_mutual_information = kwargs.get("recompute_mutual_information", True)
        recompute_loss = kwargs.get("recompute_loss", False)
        checkpoint_path = kwargs.get("checkpoint_path", None)
        reembed_checkpoint = kwargs.get("reembed_checkpoint", None)
        batch_size_inference = kwargs.get("batch_size_inference", None)
        seed = kwargs.get("seed", 42)
        path_to_data_dir = kwargs.get("path_to_data_dir", None)

        print(dataset, size, quality, algo, max_epochs, early_stopping_patience, device)
        
        if path_to_data_dir is None:
            path_to_data_dir = Path("/home/jupyter/igor_repos/noise_scaling_laws/data")
        else:
            path_to_data_dir = Path(path_to_data_dir)
        
        single_job_path = kwargs.get("single_job_path", None)
        if single_job_path is None:
            single_job_path = Path(__file__).parent.parent.parent.parent.parent / "single_job.py"
        else:
            single_job_path = Path(single_job_path)

        self.cmd = [
            "python",
            str(single_job_path),
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
            "--recompute_loss",
            str(recompute_loss).lower(),
            "--seed",
            str(seed),
        ]

        if max_steps is not None:
            self.cmd.extend(["--max_steps", str(max_steps)])

        if signal_columns:
            self.cmd.extend(["--signal_columns"] + signal_columns)

        if checkpoint_path:
            self.cmd.extend(["--checkpoint_path", str(checkpoint_path)])

        if reembed_checkpoint:
            self.cmd.extend(["--reembed_checkpoint", str(reembed_checkpoint)])

        if batch_size_inference:
            self.cmd.extend(["--batch_size_inference", str(batch_size_inference)])

        log_dir = kwargs.get("log_dir", None)
        if log_dir is not None:
            self.log_path = Path(log_dir) / f"{algo}_{size}_{quality}.txt"
        else:
            self.log_path = None

        self.job_info = {
            "dataset": dataset,
            "size": size,
            "quality": quality,
            "algo": algo,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "device": device,
            "seed": seed,
        }

    def __call__(self):
        try:
            import pty
            import sys as _sys

            ji = self.job_info
            tag = f"[{ji['dataset']},{ji['quality']},{ji['size']},{ji['algo']}]"

            if self.log_path is not None:
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "TERM": "xterm", "COLUMNS": "120"}
                with open(self.log_path, "w") as log_f:
                    header = "=" * 60 + "\n"
                    for k, v in ji.items():
                        header += f"{k}: {v}\n"
                    header += "=" * 60 + "\n\n"
                    log_f.write(header)
                    log_f.flush()
                    print(header, flush=True)

                    # Use a pseudo-TTY so Lightning shows progress bars
                    master_fd, slave_fd = pty.openpty()
                    proc = subprocess.Popen(
                        self.cmd, stdout=slave_fd, stderr=subprocess.STDOUT,
                        env=env, close_fds=True,
                    )
                    os.close(slave_fd)

                    import re

                    buf = ""
                    cur_epoch, total_epochs = 0, ji.get("max_epochs", 10)
                    t_start = time.time()

                    while True:
                        try:
                            chunk = os.read(master_fd, 8192)
                        except OSError:
                            break
                        if not chunk:
                            break
                        decoded = chunk.decode("utf-8", errors="replace")
                        log_f.write(decoded)
                        log_f.flush()

                        # Split on \n and \r to capture progress bar updates
                        buf += decoded
                        parts = re.split(r"[\r\n]", buf)
                        buf = parts[-1]  # keep incomplete tail
                        for part in parts[:-1]:
                            part = part.strip()
                            if not part:
                                continue

                            # Track epoch from progress bar lines
                            m = re.match(r"Epoch\s+(\d+)/(\d+)", part)
                            if m:
                                cur_epoch = int(m.group(1))
                                total_epochs = int(m.group(2))

                            # Skip per-batch updates, only show key lines
                            is_epoch_batch = "Epoch" in part and "it/s" in part
                            if is_epoch_batch:
                                continue

                            if any(kw in part for kw in ("Epoch", "Metric", "val_loss", "Training complete",
                                                          "Embeddings", "MI for", "Error", "Traceback",
                                                          "success!", "CUDA", "OutOfMemory")):
                                # Build epoch progress + ETA
                                elapsed = time.time() - t_start
                                if cur_epoch > 0:
                                    eta_s = elapsed / cur_epoch * (total_epochs - cur_epoch)
                                    eta = f"{eta_s/60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
                                else:
                                    eta = "?"
                                progress = f"epoch {cur_epoch}/{total_epochs} ETA {eta}"
                                _sys.stdout.write(f"{tag} ({progress}) {part}\n")
                                _sys.stdout.flush()

                    os.close(master_fd)
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, self.cmd)
            else:
                subprocess.run(self.cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {self.cmd}")
            print(f"Error: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
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
