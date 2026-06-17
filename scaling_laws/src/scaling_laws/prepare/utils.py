import os
import numpy as np
import anndata as ad
import scipy.sparse as sp
from tqdm import tqdm
import crick
import numba


def compute_medians(input_h5ad_path: str):
    file_path = input_h5ad_path
    result_genes = set()

    try:
        print(f"Reading {file_path}")
        adata = ad.read_h5ad(file_path)

        print("converting to csc")
        if sp.issparse(adata.X):
            if not isinstance(adata.X, sp.csc_matrix):
                adata.X = adata.X.tocsc()
        elif sp.issparse(adata.X[:, :]):
            adata.X = adata.X[:, :].tocsc()

        coding_miRNA_genes = adata.var.index
        coding_miRNA_loc = np.arange(len(coding_miRNA_genes))

        result_genes.update(coding_miRNA_genes)

        median_digests = [crick.tdigest.TDigest() for _ in range(len(coding_miRNA_loc))]

        if "n_counts" not in adata.obs:
            raise ValueError("n_counts not found in adata.obs")

        for i, gene_idx in tqdm(
            enumerate(coding_miRNA_loc), total=len(coding_miRNA_loc), desc="Processing genes", leave=False
        ):
            gene_data = adata.X[:, gene_idx].toarray().flatten()
            normalized_data = gene_data / adata.obs["n_counts"].values * 10_000

            if np.issubdtype(normalized_data.dtype, np.integer):
                normalized_data = normalized_data.astype(np.float32)

            nonzero_data = np.ma.masked_equal(normalized_data, 0.0).filled(np.nan)
            median_digests[i].update(nonzero_data[~np.isnan(nonzero_data)])

        digest_dict = dict(zip(coding_miRNA_genes, median_digests))
        return input_h5ad_path, digest_dict, result_genes

    except Exception as e:
        print(f"Error processing {input_h5ad_path}: {e}")
        return input_h5ad_path, {}, result_genes


@numba.njit(cache=True)
def _downsample_array(col, quality):
    cumcounts = col.cumsum()
    total = cumcounts[-1]
    target = int(total * quality)

    sample = np.random.choice(np.int_(total), target, replace=False)
    sample.sort()

    col[:] = 0
    geneptr = 0
    for count in sample:
        while count >= cumcounts[geneptr]:
            geneptr += 1
        col[geneptr] += 1
    return col


@numba.njit(parallel=True, fastmath=True)
def compute_sums(data, indices, indptr, n_rows: int, n_cols: int):
    sums = np.zeros(n_rows, dtype=np.float32)
    for i in numba.prange(n_rows):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        sums[i] = np.sum(data[row_start:row_end])
    return sums


@numba.njit(parallel=True, fastmath=True)
def downsample_matrix(data, indices, indptr, n_cols, ratio):
    for col_idx in numba.prange(n_cols):
        col_start = indptr[col_idx]
        col_end = indptr[col_idx + 1]
        if col_end > col_start:
            col_data = data[col_start:col_end]
            data[col_start:col_end] = _downsample_array(col_data, ratio)


def merge_digest(dict_key_ensembl_id, dict_value_tdigest, new_tdigest_dict):
    new_gene_tdigest = new_tdigest_dict.get(dict_key_ensembl_id)
    if new_gene_tdigest is not None:
        dict_value_tdigest.merge(new_gene_tdigest)
        return dict_value_tdigest
    elif new_gene_tdigest is None:
        return dict_value_tdigest
