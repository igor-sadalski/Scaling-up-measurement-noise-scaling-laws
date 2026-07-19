"""Standalone estimate of the auxiliary MI carried by a random orthonormal
projection of PBMC CITE-seq gene expression
It prints a single number (the auxiliary MI) and produces no figures.
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from latentmi import lmi

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

N_COMPONENTS = 16
N_HVG = 750
N_SUBSAMPLE = 20_000
SEED = 42
LMI_KWARGS = dict(validation_split=0.3, batch_size=512, epochs=300, quiet=False)


def load_pbmc_cite_seq():
    """Download PBMC CITE-seq and return (X_rna_lognorm_hvg, Y_protein).

    Preprocessing of the RNA matches the PCA pipeline: normalize_total to 1e4,
    log1p, then restrict to the top 750 highly variable genes. The protein
    counts are the auxiliary signal Y, used as-is (as in the pipeline)."""
    adata = scvi.data.pbmc_seurat_v4_cite_seq(apply_filters=True, aggregate_proteins=True)

    # Y: surface protein abundances (the "second half" of the CITE-seq measurement).
    Y = np.asarray(adata.obsm["protein_counts"].values, dtype=np.float64)

    # X: RNA, preprocessed exactly as the PCA pipeline does.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    adata = adata[:, adata.var["highly_variable"]]
    X = adata.X
    return X, Y


def random_orthonormal_projection(X, n_components, seed):
    """Project X (n_cells x n_genes) onto a random orthonormal frame of
    `n_components` columns obtained by QR of a Gaussian matrix."""
    n_genes = X.shape[1]
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_genes, n_components))
    Q, _ = np.linalg.qr(G)  # Q: (n_genes, n_components), orthonormal columns
    Z = X @ Q  # sparse (n_cells x n_genes) @ dense -> dense (n_cells x n_components)
    return np.asarray(Z, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--n_subsample", type=int, default=N_SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for the LMI estimator.")

    X, Y = load_pbmc_cite_seq()
    print(f"RNA X: {X.shape}, protein Y: {Y.shape}")

    Z = random_orthonormal_projection(X, args.n_components, args.seed)
    print(f"random orthonormal projection Z: {Z.shape}")

    # Subsample paired points before the estimate.
    n = Z.shape[0]
    if n > args.n_subsample:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.n_subsample, replace=False)
        Z, Y = Z[idx], Y[idx]
    print(f"estimating MI on {Z.shape[0]} points")

    pmi, _, _ = lmi.estimate(Z, Y, **LMI_KWARGS)
    mi = float(np.nanmean(pmi))

    print("=" * 60)
    print(f"Auxiliary MI of random orthonormal projection (PBMC, {args.n_components}d): {mi:.4f} bits")
    print("=" * 60)

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "rp_random_orthonormal_mi.csv")
    pd.DataFrame([dict(
        dataset="PBMC", signal="protein_counts", n_components=args.n_components,
        n_hvg=N_HVG, n_cells=int(Z.shape[0]), seed=args.seed, mi_bits=mi,
    )]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
