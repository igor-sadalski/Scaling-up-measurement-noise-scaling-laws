"""Build per-fold train/val h5ad splits with per-run energy distances.

Input  : per-ID h5ads recovered from old 16-fold LOO val dirs
Output : kfold/{k+1}-fold/train/{run}_edist{value}.h5ad   (8 train h5ads per fold,
         one per author_experimental_id; filename encodes the energy distance
         between that single run and the val pool, computed
         on a PCA fit jointly on train+val for the fold)
         kfold/{k+1}-fold/val/{run}.h5ad                  (8 val h5ads per fold)
         kfold/folds.json                                  (index of all folds)

8-fold CV across the 16 IDs: each fold holds 8 IDs (50%) out for validation
and trains on the remaining 8. Val sets are independently sampled per fold so
overlap between folds is allowed. Energy distance for a train run uses geomloss
on a 5k subsample of PC scores from a PCA fit on the full train+val concatenation
for that fold.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from geomloss import SamplesLoss
from tqdm.auto import tqdm

# Per-ID h5ads are recovered from the old 16-fold LOO val directories.
# Each ID appeared as the single val set in exactly one old fold, so all 16
# canonical per-ID h5ads are recoverable without the original source h5ad.
OLD_KFOLD_SRC = Path(
    "/home/igor/igor_repos/scaling_laws/data_local/old/batch_effects/kfold"
)
DST_DIR = Path("/home/igor/igor_repos/scaling_laws/data_local/other/batch_effects/kfold")
FOLDS_PATH = DST_DIR / "folds.json"
LOG_PATH = DST_DIR / "2026-04-27_build_kfold_splits.log"

COL = "author_experimental_id"
N_FOLDS = 8
VAL_PER_FOLD = 8
SEED = 0
N_PCS = 50
EDIST_SAMPLES = 5000
EXPECTED_PER_ID = 10_000

EXPECTED_IDS = [
    "run_4", "run_13", "run_14", "run_15", "run_16", "run_17", "run_18",
    "run_19", "run_20", "run_21", "run_22", "run_23", "run_24", "run_25",
    "run_26", "run_27",
]


def fold_dir(k: int) -> Path:
    """1-indexed fold directory: k=0 -> 1-fold, k=1 -> 2-fold, ..."""
    return DST_DIR / f"{k + 1}-fold"


class TqdmToLogger(io.StringIO):
    """Pipe tqdm bar updates into the logger so they end up in the log file too."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf: str) -> int:  # type: ignore[override]
        self.buf = buf.strip("\r\n\t ")
        return len(buf)

    def flush(self) -> None:  # type: ignore[override]
        if self.buf:
            self.logger.log(self.level, self.buf)
            self.buf = ""


def setup_logging() -> logging.Logger:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("build_kfold_splits")
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


def fold_pcs(
    train_adatas: list[ad.AnnData],
    val_adatas: list[ad.AnnData],
    n_pcs: int = N_PCS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on log-normalized train+val and return (train_pcs, val_pcs)."""
    combined = ad.concat(train_adatas + val_adatas, join="inner")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.pca(combined, n_comps=n_pcs, random_state=SEED)
    pcs = np.asarray(combined.obsm["X_pca"])
    n_train = sum(a.n_obs for a in train_adatas)
    return pcs[:n_train], pcs[n_train:]


def energy_distance(
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int = EDIST_SAMPLES,
    seed: int = SEED,
) -> float:
    """geomloss SamplesLoss(loss='energy') on subsamples to bound memory."""
    rng = np.random.default_rng(seed)
    ai = rng.choice(len(a), size=min(n_samples, len(a)), replace=False)
    bi = rng.choice(len(b), size=min(n_samples, len(b)), replace=False)
    loss = SamplesLoss(loss="energy")
    at = torch.from_numpy(np.ascontiguousarray(a[ai])).float()
    bt = torch.from_numpy(np.ascontiguousarray(b[bi])).float()
    return float(loss(at, bt).item())


def build_folds(ids: list[str]) -> list[dict]:
    """Independently sample VAL_PER_FOLD val IDs per fold; overlaps allowed."""
    folds = []
    for k in range(N_FOLDS):
        rng_k = np.random.default_rng(SEED + k)
        val_ids = sorted(
            rng_k.choice(ids, size=VAL_PER_FOLD, replace=False).tolist()
        )
        train_ids = sorted([i for i in ids if i not in val_ids])
        folds.append({"fold": k, "val_ids": val_ids, "train_ids": train_ids})
    return folds


def load_per_id_from_old_kfold(log: logging.Logger) -> tuple[dict[str, ad.AnnData], dict[str, int]]:
    """Recover per-ID h5ads from old LOO val dirs (each ID is one fold's val)."""
    per_id: dict[str, ad.AnnData] = {}
    per_id_counts: dict[str, int] = {}
    for fold_path in sorted(OLD_KFOLD_SRC.glob("*-fold")):
        val_dir = fold_path / "val"
        for h5ad_path in sorted(val_dir.glob("*.h5ad")):
            run_id = h5ad_path.stem
            if run_id in per_id:
                continue  # already loaded
            if run_id not in EXPECTED_IDS:
                log.warning("unexpected id %s in %s — skipping", run_id, h5ad_path)
                continue
            t0 = time.time()
            per_id[run_id] = ad.read_h5ad(h5ad_path)
            per_id_counts[run_id] = int(per_id[run_id].n_obs)
            log.info("  %-8s n=%d loaded from %s (%.1f s)",
                     run_id, per_id_counts[run_id],
                     h5ad_path.relative_to(OLD_KFOLD_SRC), time.time() - t0)

    missing = set(EXPECTED_IDS) - set(per_id)
    if missing:
        raise FileNotFoundError(
            f"could not find h5ads for ids: {sorted(missing)}\n"
            f"searched under {OLD_KFOLD_SRC}"
        )
    return per_id, per_id_counts


def main() -> None:
    log = setup_logging()
    log.info("old kfold source: %s", OLD_KFOLD_SRC)
    log.info("destination: %s", DST_DIR)

    existing_fold_dirs = sorted(DST_DIR.glob("*-fold"))
    if existing_fold_dirs:
        raise SystemExit(
            f"refusing to start: {DST_DIR} already contains fold subtrees: "
            f"{[p.name for p in existing_fold_dirs]}. Move them aside (e.g. into "
            f"{DST_DIR.parent / 'old'}) before rebuilding."
        )

    log.info("loading per-ID h5ads from old kfold val dirs ...")
    per_id, per_id_counts = load_per_id_from_old_kfold(log)
    log.info("loaded %d IDs", len(per_id))

    folds = build_folds(EXPECTED_IDS)

    expected_train_total = (len(EXPECTED_IDS) - VAL_PER_FOLD) * EXPECTED_PER_ID
    expected_val_total = VAL_PER_FOLD * EXPECTED_PER_ID

    tqdm_log = TqdmToLogger(log)
    for f in tqdm(folds, desc="building folds", file=tqdm_log, mininterval=1.0):
        out = fold_dir(f["fold"])
        train_out = out / "train"
        val_out = out / "val"
        train_out.mkdir(parents=True)
        val_out.mkdir(parents=True)

        train_total = sum(per_id_counts[i] for i in f["train_ids"])
        val_total = sum(per_id_counts[i] for i in f["val_ids"])
        log.info(
            "fold %d (-> %s): val=%s (n=%d) train(n_ids=%d, n=%d)=%s",
            f["fold"], out.name, f["val_ids"], val_total,
            len(f["train_ids"]), train_total, f["train_ids"],
        )
        if train_total < expected_train_total:
            log.warning(
                "fold %d train has %d cells, expected %d — short by %d",
                f["fold"], train_total, expected_train_total,
                expected_train_total - train_total,
            )
        if val_total < expected_val_total:
            log.warning(
                "fold %d val has %d cells, expected %d — short by %d",
                f["fold"], val_total, expected_val_total,
                expected_val_total - val_total,
            )

        # Write val h5ads; one per id, no edist in name.
        val_files: list[str] = []
        for vid in f["val_ids"]:
            v_path = val_out / f"{vid}.h5ad"
            t0 = time.time()
            per_id[vid].write_h5ad(v_path, compression="gzip")
            log.info(
                "  val  %-8s -> %s (%.2f MB, %.1f s)",
                vid, v_path.relative_to(DST_DIR), v_path.stat().st_size / 1e6,
                time.time() - t0,
            )
            val_files.append(str(v_path.relative_to(DST_DIR)))

        # Compute joint PCA over (8 train + 8 val) for this fold and reuse for
        # all 8 per-run energy-distance computations.
        train_adatas = [per_id[i] for i in f["train_ids"]]
        val_adatas = [per_id[i] for i in f["val_ids"]]
        t0 = time.time()
        train_pcs, val_pcs = fold_pcs(train_adatas, val_adatas)
        log.info(
            "  PCA(%d) fit on train+val (n=%d, sub=%d) in %.1f s",
            N_PCS, train_pcs.shape[0] + val_pcs.shape[0],
            min(EDIST_SAMPLES, train_pcs.shape[0], val_pcs.shape[0]),
            time.time() - t0,
        )

        # Per-run energy distance (this run's PCs vs concatenated val PCs).
        train_records: list[dict] = []
        offsets = np.cumsum([0] + [a.n_obs for a in train_adatas])
        for j, tid in enumerate(f["train_ids"]):
            run_pcs = train_pcs[offsets[j]:offsets[j + 1]]
            t0 = time.time()
            edist = energy_distance(run_pcs, val_pcs)
            t_path = train_out / f"{tid}_edist{edist:.4f}.h5ad"
            t1 = time.time()
            per_id[tid].write_h5ad(t_path, compression="gzip")
            log.info(
                "  train %-8s edist=%.4f (%.1f s) -> %s (%.2f MB, %.1f s)",
                tid, edist, t1 - t0, t_path.relative_to(DST_DIR),
                t_path.stat().st_size / 1e6, time.time() - t1,
            )
            train_records.append(
                {
                    "id": tid,
                    "edist": edist,
                    "file": str(t_path.relative_to(DST_DIR)),
                }
            )

        f["val_files"] = val_files
        f["train"] = train_records
        f["n_train"] = int(train_total)
        f["n_val"] = int(val_total)
        del train_adatas, val_adatas, train_pcs, val_pcs

    folds_serialized = []
    for f in folds:
        folds_serialized.append({
            "fold": f["fold"],
            "val_ids": f["val_ids"],
            "val_files": f["val_files"],
            "train": f["train"],
            "n_train": f["n_train"],
            "n_val": f["n_val"],
        })

    cfg = {
        "source": str(OLD_KFOLD_SRC),
        "id_column": COL,
        "n_folds": N_FOLDS,
        "val_per_fold": VAL_PER_FOLD,
        "seed": SEED,
        "n_pcs": N_PCS,
        "edist_subsample": EDIST_SAMPLES,
        "expected_per_id": EXPECTED_PER_ID,
        "per_id_counts": per_id_counts,
        "all_ids": list(EXPECTED_IDS),
        "folds": folds_serialized,
    }
    FOLDS_PATH.write_text(json.dumps(cfg, indent=2))
    log.info("wrote %s", FOLDS_PATH)
    log.info("done.")


if __name__ == "__main__":
    main()
