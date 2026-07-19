"""
Geneformer's tokenization reduces per-cell abundances to rank (order-statistic)
information. This script quantifies the resulting information loss in a
model-free way: it takes the same HVG-selected, log-normalized RNA as
`rp_random_orthonormal_mi.py`, and estimates I(Z; protein) for two inputs
projected through the *same* random orthonormal frame:

  1. continuous    : the log-normalized expression (as before);
  2. order-statistic: each cell's expression rank-transformed across genes.
"""

import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
import torch
from latentmi import lmi

from rp_random_orthonormal_mi import (
    FIGDIR,
    LMI_KWARGS,
    N_COMPONENTS,
    N_HVG,
    N_SUBSAMPLE,
    SEED,
    load_pbmc_cite_seq,
    random_orthonormal_projection,
)


def to_order_statistics(X):
    """Rank-transform each cell (row) across genes, emulating the rank-value
    encoding used by Geneformer's tokenizer. Ties get the average rank."""
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    return rankdata(X_dense, axis=1).astype(np.float64)


def estimate_mi(Z, Y):
    pmi, _, _ = lmi.estimate(Z, Y, **LMI_KWARGS)
    return float(np.nanmean(pmi))


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

    X_ord = to_order_statistics(X)

    # Same random orthonormal frame for both inputs (same seed, same n_genes),
    # so any difference in MI is attributable to the order-statistic transform.
    Z_cont = random_orthonormal_projection(X, args.n_components, args.seed)
    Z_ord = random_orthonormal_projection(X_ord, args.n_components, args.seed)

    # Subsample the same paired points for both estimates.
    n = Z_cont.shape[0]
    if n > args.n_subsample:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.n_subsample, replace=False)
        Z_cont, Z_ord, Y = Z_cont[idx], Z_ord[idx], Y[idx]
    print(f"estimating MI on {Z_cont.shape[0]} points")

    mi_cont = estimate_mi(Z_cont, Y)
    mi_ord = estimate_mi(Z_ord, Y)

    print("=" * 60)
    print(f"continuous     random orthonormal projection MI: {mi_cont:.4f} bits")
    print(f"order-statistic random orthonormal projection MI: {mi_ord:.4f} bits")
    print(f"information loss from order-statistic transform : {mi_cont - mi_ord:.4f} bits")
    print("=" * 60)

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "rp_order_statistic_mi.csv")
    pd.DataFrame([dict(
        dataset="PBMC", signal="protein_counts", n_components=args.n_components,
        n_hvg=N_HVG, n_cells=int(Z_cont.shape[0]), seed=args.seed,
        mi_continuous_bits=mi_cont, mi_order_statistic_bits=mi_ord,
        mi_loss_bits=mi_cont - mi_ord,
    )]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
