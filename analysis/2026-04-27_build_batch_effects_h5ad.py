"""Sample 10k cells per author_experimental_id from the shendure raw h5ad.

This version loads the entire source h5ad into memory up-front, then performs
sampling and writing. Requires ~230-400 GB of free RAM (CSR X has ~19.5B nnz).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from tqdm.auto import tqdm

SRC = Path(
    "/home/igor/igor_repos/scaling_laws/data/shendure/774263/1.0/preprocessed/preprocessed.h5ad"
)
DST_DIR = Path("/home/igor/igor_repos/scaling_laws/data_local/other/batch_effects")
DST = DST_DIR / "shendure_774263_q1.0_preprocessed_author_experimental_id_10k_per_cat.h5ad"
LOG_PATH = Path(__file__).resolve().parent / "2026-04-27_build_batch_effects_h5ad.log"

PER_CATEGORY = 10_000
SEED = 0
COL = "author_experimental_id"


def setup_logging() -> logging.Logger:
    if not DST_DIR.exists():
        DST_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("build_batch_effects")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def main() -> None:
    log = setup_logging()
    rng = np.random.default_rng(SEED)
    log.info("source: %s (%.2f GB on disk)", SRC, SRC.stat().st_size / 1e9)
    log.info("destination: %s", DST)

    log.info("loading full h5ad into memory (this can take a while)...")
    t0 = time.time()
    adata = ad.read_h5ad(SRC)
    log.info("loaded in %.1f s. shape=%s, X dtype=%s, X nnz=%s",
             time.time() - t0, adata.shape, adata.X.dtype,
             getattr(adata.X, "nnz", "n/a"))

    if COL not in adata.obs.columns:
        raise KeyError(f"{COL!r} not in obs columns: {list(adata.obs.columns)}")

    labels = adata.obs[COL].to_numpy()
    categories = (
        adata.obs[COL].cat.categories.tolist()
        if hasattr(adata.obs[COL], "cat")
        else list(np.unique(labels))
    )
    log.info("found %d categories in %s", len(categories), COL)

    selected: list[np.ndarray] = []
    for cat in tqdm(categories, desc="sampling per category"):
        idx = np.flatnonzero(labels == cat)
        take = min(PER_CATEGORY, idx.size)
        chosen = rng.choice(idx, size=take, replace=False) if take < idx.size else idx
        selected.append(chosen)
        log.info("  cat=%-24s available=%-9d sampled=%d", cat, idx.size, take)

    keep = np.sort(np.concatenate(selected)).astype(np.int64)
    log.info("total selected rows: %d", keep.size)

    log.info("subsetting AnnData ...")
    t0 = time.time()
    subset = adata[keep].copy()
    log.info("subset done in %.1f s. shape=%s", time.time() - t0, subset.shape)

    log.info("freeing source AnnData ...")
    del adata

    log.info("writing %s ...", DST)
    t0 = time.time()
    subset.write_h5ad(DST, compression="gzip")
    log.info("wrote in %.1f s. output size: %.2f GB",
             time.time() - t0, DST.stat().st_size / 1e9)
    log.info("done.")


if __name__ == "__main__":
    main()
