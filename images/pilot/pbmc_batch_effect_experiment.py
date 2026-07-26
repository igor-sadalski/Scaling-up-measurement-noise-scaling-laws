#!/mnt/home/gokulg/miniconda3/envs/lt/bin/python
#SBATCH --job-name=pbmc_batch_effect
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.err
#
# pbmc_batch_effect_experiment.py
#
# CITE-seq analogue of batch_effect_experiment.py. Signal Y = surface protein counts;
# representation Z = SCVI latent of the RNA counts; MI = lmi.estimate(Z, Y) (bits).
# 2-D sweep over measurement quality (UMI downsampling) x batch strength sigma_M.
#   measurement noise: binomial thinning of counts, fraction q  (eta := q, sigma_meas^2 := 1/q)
#   batch effect:      fixed per-batch, per-gene MULTIPLICATIVE factor exp(sigma_M * N(0,1)),
#                      coherent across cells in a batch, uncorrected (no batch_key to SCVI).
#                      The factor is log-normal (log-additive == count-multiplicative), hence
#                      strictly positive, so corrupted counts stay >= 0. Biologically this
#                      reads as a batch-specific per-gene fold-change (capture / efficiency)
#                      scaling the true expression, applied BEFORE the UMI thinning.
# SCVI config + CITE-seq loader follow scaling_laws/algo/scvi.py and
# analysis/rebuttal_figures/rp_random_orthonormal_mi.py.
#
# sigma_M indexes the log-std of the per-gene multiplicative factor: at sigma_M=1 the factor
# spans exp([-3,3]) ~ [0.05, 20]x (clip=3), coherent across a batch, so sigma_M in [0,1]
# spans negligible-to-strong batch corruption. sigma_meas^2 ~ 1/q is kept as the
# measurement-noise axis (set by UMI depth q).

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # shim: latentmi still uses the np.NaN alias removed in numpy 2.0
import pandas as pd
import scanpy as sc
import scvi
import torch

PILOT_DIR = Path('/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot')
if not PILOT_DIR.exists():
    PILOT_DIR = Path(__file__).resolve().parent
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from latentmi import lmi  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
log = logging.getLogger('pbmc_batch_effect')


def load_pbmc(n_hvg, n_cells, seed):
    """Raw RNA counts (top-n_hvg HVGs) + protein-count signal Y, subsampled to n_cells.

    Also returns the full-depth mean UMI/cell over the HVGs (computed before subsampling) --
    the clean measurement-depth scale used as eta = q * mean_umi_full in the analysis."""
    adata = scvi.data.pbmc_seurat_v4_cite_seq(apply_filters=True, aggregate_proteins=True)
    Y = np.asarray(adata.obsm['protein_counts'].values, dtype=np.float64)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')  # on raw counts
    adata = adata[:, adata.var['highly_variable']].copy()
    X = np.asarray(adata.X.todense() if hasattr(adata.X, 'todense') else adata.X)
    X = np.rint(X).astype(np.int64)                                            # integer counts
    mean_umi_full = float(X.sum(axis=1).mean())                                # full-depth, over HVGs
    if n_cells and n_cells < X.shape[0]:
        idx = np.random.default_rng(seed + 12345).choice(X.shape[0], n_cells, replace=False)
        X, Y = X[idx], Y[idx]
    return X, Y, mean_umi_full


def batch_offsets(seed, k, n_genes, sigma_M, clip=3.0):
    """Fixed per-(batch, gene) multiplicative factors exp(sigma_M * N(0,1)); None if sigma_M=0."""
    if sigma_M == 0:
        return None
    z = np.random.default_rng(seed + 777).standard_normal((k, n_genes))
    return np.exp(sigma_M * np.clip(z, -clip, clip)).astype(np.float64)


def corrupt(X, batch_of_cell, offsets, quality, rng):
    """Scale by the coherent log-normal batch factor (fixed), then binomial-thin to fraction quality."""
    Xc = X
    if offsets is not None:
        # multiplicative, strictly-positive factor -> corrupted counts stay >= 0
        Xc = np.rint(X * offsets[batch_of_cell]).astype(np.int64)
    if quality < 1.0:
        Xc = rng.binomial(Xc, quality)
    return Xc


def scvi_embed(X, max_epochs, n_latent, seed):
    """Train SCVI on counts X and return the latent representation (scaling_laws config)."""
    scvi.settings.seed = seed
    adata = sc.AnnData(X.astype(np.float32))
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata, n_hidden=512, n_latent=n_latent, n_layers=1,
                          dropout_rate=0.1, dispersion='gene',
                          gene_likelihood='zinb', latent_distribution='normal')
    vae.train(accelerator='gpu', devices=1, train_size=0.8, validation_size=0.2,
              batch_size=512, max_epochs=max_epochs, early_stopping=True,
              early_stopping_monitor='elbo_validation', early_stopping_patience=5,
              enable_progress_bar=False)
    return vae.get_latent_representation(adata)


def run_cell(X, Y, quality, sigma_M, seed, args):
    """Corrupt -> SCVI embed -> MI(Z; protein counts). Returns MI in bits."""
    n_cells, n_genes = X.shape
    split_rng = np.random.default_rng(seed)
    batch_of_cell = split_rng.integers(0, args.k, size=n_cells)
    offsets = batch_offsets(seed, args.k, n_genes, sigma_M)
    Xc = corrupt(X, batch_of_cell, offsets, quality, np.random.default_rng(seed + 1))

    # At the very low depths used to extend the baseline, thinning yields some zero-count
    # cells whose (log) library size is undefined and breaks SCVI. Drop them from Xc and the
    # aligned signal Y. No-op at the original depths (no zero-count cells there).
    Yc = Y
    nz = Xc.sum(axis=1) > 0
    if not nz.all():
        log.info("  q=%.5f sigma_M=%.3f: dropping %d/%d zero-count cells",
                 quality, sigma_M, int((~nz).sum()), len(nz))
        Xc, Yc = Xc[nz], Y[nz]

    log.info("cell q=%.4f sigma_M=%.3f: SCVI (%d cells x %d genes, %d epochs)",
             quality, sigma_M, Xc.shape[0], n_genes, args.max_epochs)
    t0 = time.time()
    Z = scvi_embed(Xc, args.max_epochs, args.n_latent, seed)
    pmi, _, _ = lmi.estimate(Z, Yc, validation_split=0.3, batch_size=512,
                             epochs=args.lmi_epochs, quiet=True)
    mi = float(np.nanmean(pmi))
    log.info("  -> MI=%.4f bits (%.1fs)", mi, time.time() - t0)
    return mi


def main():
    p = argparse.ArgumentParser(description="PBMC CITE-seq batch-effect MI sweep (SCVI).")
    p.add_argument('--output-dir', default=str(PILOT_DIR / 'results_pbmc_mult'))
    p.add_argument('--n-hvg', type=int, default=750)
    p.add_argument('--n-cells', type=int, default=20000)
    p.add_argument('--qualities', type=float, nargs='+', default=[1.0, 0.5, 0.25, 0.125, 0.0625],
                   help='UMI keep-fraction grid (1.0 = full depth = least noise)')
    p.add_argument('--sigma-M', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help='batch strength: log-std of multiplicative log-normal per-gene factor (0 = baseline)')
    p.add_argument('--baseline-extra-qualities', type=float, nargs='*',
                   default=[0.03125, 0.015625, 0.0078125, 0.00390625],
                   help='extra UMI keep-fractions run ONLY at sigma_M=0, to extend the baseline '
                        'curve down to smaller mean UMI/cell (so strong-batch curves collapse onto '
                        'measured, not extrapolated, baseline). Pass with no values to disable.')
    p.add_argument('--append', action='store_true',
                   help='load an existing results.csv and skip already-computed (seed, quality, '
                        'sigma_M) cells; only new cells are run and appended')
    p.add_argument('--k', type=int, default=20, help='number of batches')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--max-epochs', type=int, default=10, help='SCVI epochs')
    p.add_argument('--lmi-epochs', type=int, default=100, help='latentmi estimator epochs')
    p.add_argument('--n-latent', type=int, default=16)
    p.add_argument('--no-analyze', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for SCVI + latentmi.")
    log.info("args: %s", vars(args))

    # (quality, sigma_M) cells: the full grid, plus baseline-only (sigma_M=0) extras at
    # smaller depth that extend the baseline curve down.
    pairs = [(float(q), float(sm)) for q in args.qualities for sm in args.sigma_M]
    pairs += [(float(q), 0.0) for q in args.baseline_extra_qualities]

    n_cells_used = None
    rows, results_csv = [], os.path.join(args.output_dir, 'results.csv')
    done = set()  # (seed, quality, sigma_M) cells already present when appending
    if args.append and os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        rows = prev.to_dict('records')
        done = {(int(r['seed']), round(float(r['quality']), 12), round(float(r['sigma_M']), 12))
                for _, r in prev.iterrows()}
        log.info("append mode: loaded %d existing rows from %s", len(rows), results_csv)

    for seed in args.seeds:
        todo = [(q, sm) for (q, sm) in pairs
                if (int(seed), round(q, 12), round(sm, 12)) not in done]
        if not todo:
            log.info("[seed %d] all cells already present; skipping", seed)
            continue
        X, Y, mean_umi_full = load_pbmc(args.n_hvg, args.n_cells, seed)
        n_cells_used = X.shape[0]
        # self-contained eta scale for the analysis (clean full-depth mean UMI/cell over HVGs)
        with open(os.path.join(args.output_dir, 'mean_umi_full.json'), 'w') as f:
            json.dump(dict(n_hvg=int(args.n_hvg), mean_umi_per_cell_hvg=mean_umi_full), f, indent=2)
        log.info("[seed %d] X=%s counts, Y=%s protein; mean UMI/cell(full)=%.1f; %d cells to run",
                 seed, X.shape, Y.shape, mean_umi_full, len(todo))
        for quality, sigma_M in todo:
            sigma_meas_sq = 1.0 / quality           # heuristic measurement-variance proxy
            eta = quality
            ratio = (sigma_M ** 2) / sigma_meas_sq
            mi = run_cell(X, Y, quality, float(sigma_M), seed, args)
            rows.append(dict(
                seed=seed, noise_level=float(quality), quality=float(quality),
                eta=float(eta), sigma_meas_sq=float(sigma_meas_sq),
                sigma_M=float(sigma_M), sigma_M_sq=float(sigma_M ** 2),
                sigma_M_sq_ratio=float(ratio), k=int(args.k), MI=mi))
            pd.DataFrame(rows).to_csv(results_csv, index=False)  # checkpoint per cell

    meta = dict(
        dataset='PBMC_CITEseq',
        classifier='SCVI (n_hidden=512, n_latent=%d, zinb)' % args.n_latent,
        optimizer='SCVI default train plan',
        mi_estimator='latentmi.lmi.estimate(Z, protein_counts) [bits]',
        signal='protein_counts',
        epochs=args.max_epochs, k=args.k, seeds=list(args.seeds),
        n_train=n_cells_used, n_test=n_cells_used, n_hvg=args.n_hvg,
        noise_grid=list(map(float, args.qualities)),
        baseline_extra_qualities=list(map(float, args.baseline_extra_qualities)),
        sigma_M_grid=list(map(float, args.sigma_M)),
    )
    with open(os.path.join(args.output_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s and run_meta.json (%d rows)", results_csv, len(rows))

    if not args.no_analyze:
        try:
            import analyze_pbmc_batch_effect
            summary = analyze_pbmc_batch_effect.generate(args.output_dir)
            log.info("analysis done:\n%s", json.dumps(summary, indent=2))
        except Exception as e:
            log.exception("analysis step failed (results.csv is intact): %s", e)
    log.info("ALL DONE.")


if __name__ == '__main__':
    main()
