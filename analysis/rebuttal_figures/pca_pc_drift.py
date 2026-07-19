"""Quantify how PCA principal components drift with sample number

Drift is the principal angle between the component and the reference component (robust to the sign ambiguity of eigenvectors); results are averaged over random subsamples.

Produces one curve per principal component. Writes a figure and a CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import utils

from rp_random_orthonormal_mi import load_pbmc_cite_seq

N_COMPONENTS = 8
SIZES = np.logspace(2, 4, 10).astype(int)
REF_SIZE = 10_000
REF_SEED = 0
SEEDS = range(1, 101)  # subsamples averaged per size (disjoint from REF_SEED)


def fit_components(X, n, seed, n_components):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    svd.fit(X[idx])
    return svd.components_  # (n_components, n_genes), unit-norm rows


def drift_angles(comps, ref):
    """Principal angle (degrees) between matched components, robust to sign."""
    cos = np.abs(np.sum(comps * ref, axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def main():
    X, _ = load_pbmc_cite_seq()
    print(f"X: {X.shape}")

    ref = fit_components(X, REF_SIZE, REF_SEED, N_COMPONENTS)

    # drift[size, pc] averaged over seeds
    rows = []
    mean_drift = np.zeros((len(SIZES), N_COMPONENTS))
    for i, n in enumerate(SIZES):
        per_seed = np.array([drift_angles(fit_components(X, n, s, N_COMPONENTS), ref)
                             for s in SEEDS])
        mean_drift[i] = per_seed.mean(axis=0)
        for pc in range(N_COMPONENTS):
            rows.append(dict(size=int(n), pc=pc + 1,
                             drift_deg=mean_drift[i, pc],
                             drift_std=per_seed[:, pc].std()))

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    cmap = plt.get_cmap("viridis")
    for pc in range(N_COMPONENTS):
        ax.plot(SIZES, mean_drift[:, pc], marker="o", ms=3, lw=1.2,
                color=cmap(pc / (N_COMPONENTS - 1)), label=f"PC{pc + 1}")
    ax.set_xscale("log")
    ax.set_xlabel("cells used to fit PCA")
    ax.set_ylabel("drift from $10^4$-cell PC (degrees)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=1)
    utils.save(fig, "09_pca_pc_drift")

    out = os.path.join(utils.FIGDIR, "09_pca_pc_drift.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    print("\nmean drift (deg) at smallest vs largest sample size:")
    for pc in range(N_COMPONENTS):
        print(f"  PC{pc + 1}: {mean_drift[0, pc]:6.1f} (n={SIZES[0]})"
              f"  ->  {mean_drift[-1, pc]:5.1f} (n={SIZES[-1]})")


if __name__ == "__main__":
    main()
