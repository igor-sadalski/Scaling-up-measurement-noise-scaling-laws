# analyze_pbmc_batch_effect.py
# PBMC-specific collapse analysis with a CONSTANT-per-curve rescale.
#
# Model (this script):   eta_eff = eta / (1 + s),   s constant per sigma_M curve.
# Motivation: if the effective SNR is eta_eff = eta / (1 + sigma_M^2 + sigma_B^2) with
# sigma_M^2 (batch) and sigma_B^2 (biological) both independent of depth, then the
# denominator is a single constant per curve -- a pure horizontal shift of the curve in
# log-eta. We fit one s per curve to collapse it onto the sigma_M=0 baseline. This differs
# from the depth-dependent form eta/(1 + alpha*sigma_M^2/sigma_meas^2) (= 1 + alpha*sigma_M^2*eta)
# used in analyze_batch_effect.py, whose denominator grows with eta.
#
# eta axis: eta = mean UMI/cell = q * (full-depth mean UMI/cell over the HVGs the model
# sees). The full-depth mean is read from results_pbmc/mean_umi_full.json (written by the
# one-off that calls the experiment's load_pbmc), or from --mean-umi-full, else eta=q with
# a note. NOTE: rescaling eta by a global constant leaves s and the collapse unchanged
# (s = eta/eta_star - 1 is scale-invariant); the UMI scaling only sets the absolute axis.
#
# s is fit per curve to minimize the MSE of the shifted curve eta/(1+s) against the MEASURED
# baseline (baseline MI read by interpolation; the baseline is not itself curve-fit, and the
# baseline is not inverted MI->eta). For this to be meaningful the baseline must be measured
# down to the eta_eff the strong curves reach -- run the experiment with
# --baseline-extra-qualities to extend it; n_below_baseline flags shortfall.
# fig3 fits s vs sigma_M^2: under the model above, collapsed onto the (biased) baseline,
# s = sigma_M^2/(1+sigma_B^2) -- a line through the origin with slope 1/(1+sigma_B^2), so a
# near-zero intercept + slope<=1 supports it and gives sigma_B^2 = 1/slope - 1.
#
# Outputs: fig1_raw_curves.png, fig2_collapse_const_s.png, fig3_s_vs_sigmaM2.png,
#          summary_const_s.json.

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(results_dir):
    df = pd.read_csv(os.path.join(results_dir, 'results.csv'))
    agg = (df.groupby(['noise_level', 'sigma_M'], as_index=False)
             .agg(MI=('MI', 'mean'),
                  MI_sem=('MI', lambda s: s.std(ddof=1) / max(1, np.sqrt(len(s))) if len(s) > 1 else 0.0),
                  q=('quality', 'first'),
                  sigma_M_sq=('sigma_M_sq', 'first'),
                  n_seeds=('MI', 'size')))
    meta_path = os.path.join(results_dir, 'run_meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return agg, meta


def resolve_eta(agg, results_dir, mean_umi_full):
    """Set the eta column to mean UMI/cell = q * full-depth mean UMI/cell.

    Precedence: --mean-umi-full arg > results_pbmc/mean_umi_full.json > eta=q (with note)."""
    note = None
    if mean_umi_full is None:
        cache = os.path.join(results_dir, 'mean_umi_full.json')
        if os.path.exists(cache):
            mean_umi_full = float(json.load(open(cache))['mean_umi_per_cell_hvg'])
        else:
            note = ('mean_umi_full.json not found and --mean-umi-full not given; '
                    'using eta=q. s and collapse are unchanged (scale-invariant); '
                    'only the eta axis differs.')
            mean_umi_full = 1.0
    agg = agg.copy()
    agg['eta'] = agg['q'].to_numpy() * mean_umi_full
    return agg, float(mean_umi_full), note


def baseline_curve(agg):
    """sigma_M == 0 curve, sorted by eta ascending."""
    b = agg[np.isclose(agg['sigma_M'], 0.0)].sort_values('eta')
    return b['eta'].to_numpy(), b['MI'].to_numpy()


def baseline_MI_at_eta(eta_query, eta_b, mi_b):
    """Baseline MI read off the measured baseline curve at eta (log-eta interp).

    np.interp clamps at the baseline's endpoints -- past the measured range it returns the
    end value rather than extrapolating a fitted line. So s should be fit against a baseline
    that has been *measured* down to the eta_eff the strong curves reach (that is what the
    experiment's --baseline-extra-qualities is for); n_below_baseline flags any point that
    still lands below the measured range."""
    order = np.argsort(eta_b)
    return np.interp(np.log(np.atleast_1d(eta_query)), np.log(eta_b[order]), mi_b[order])


def curve(agg, s):
    """One sigma_M curve (eta ascending): eta, MI arrays."""
    sub = agg[np.isclose(agg['sigma_M'], s)].sort_values('eta')
    return sub['eta'].to_numpy(), sub['MI'].to_numpy()


def fit_s_curve(eta, mi, eta_b, mi_b, s_grid):
    """Fit one constant s for eta_eff = eta/(1+s) minimizing the MSE of the shifted curve
    vs the baseline: residual_i = MI_i - baseline_MI_at_eta(eta_i/(1+s)).

    The baseline MI is read off the measured baseline by interpolation (no inversion / no
    curve fit to the baseline). All points contribute at every s (none are dropped), so the
    objective does not reward pushing points out of range. Reports the best s, its MSE,
    whether it sits at a grid edge, and how many points land below the measured baseline
    (which would mean the baseline needs extending further down)."""
    def mse(s):
        resid = mi - baseline_MI_at_eta(eta / (1.0 + s), eta_b, mi_b)
        return float(np.mean(resid ** 2))
    mses = np.array([mse(s) for s in s_grid])
    j = int(np.argmin(mses))
    s_hat = float(s_grid[j])
    n_below = int(np.sum(eta / (1.0 + s_hat) < eta_b.min()))
    return dict(s=s_hat,
                residual_mse=float(mses[j]),
                at_grid_edge=bool(j == 0 or j == len(s_grid) - 1),
                n_below_baseline=n_below,
                n_points=int(len(eta)))


def sigma_M_palette(sigma_vals):
    cmap = plt.get_cmap('viridis')
    return {s: cmap(i / max(1, len(sigma_vals) - 1)) for i, s in enumerate(sigma_vals)}


def fig1_raw(agg, colors, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in sorted(agg['sigma_M'].unique()):
        sub = agg[np.isclose(agg['sigma_M'], s)].sort_values('eta')
        is_base = np.isclose(s, 0.0)
        ax.plot(sub['eta'], sub['MI'], marker='o', ms=4,
                color='k' if is_base else colors[s],
                lw=2.4 if is_base else 1.6,
                label='baseline' if is_base else f"$\\sigma_M$={s:.3f}")
    ax.set_xscale('log')
    ax.set_xlabel(r'$\eta$ = mean UMI/cell')
    ax.set_ylabel('mi (bits)')
    ax.set_title('raw mi curves (pbmc)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_collapse(agg, colors, fits, out):
    """MI vs eta_eff = eta/(1+s), each curve using its own fitted s. Overlays baseline."""
    eta_b, mi_b = baseline_curve(agg)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(eta_b, mi_b, color='k', lw=2.4, marker='o', ms=4, zorder=5, label='baseline')
    for s in sorted(agg['sigma_M'].unique()):
        if np.isclose(s, 0.0):
            continue
        eta, mi = curve(agg, s)
        s_hat = fits[s]['s']
        eta_eff = eta / (1.0 + s_hat)
        ax.plot(eta_eff, mi, marker='s', ms=4, lw=1.4, color=colors[s],
                label=f"$\\sigma_M$={s:.3f} ($s$={s_hat:.2f})")
    ax.set_xscale('log')
    ax.set_xlabel(r'$\eta_{\mathrm{eff}} = \eta/(1+s)$   (mean UMI/cell)')
    ax.set_ylabel('mi (bits)')
    ax.set_title('constant-$s$ collapse (pbmc)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3_s_vs_sigmaM2(fits, colors, out):
    """Fitted s vs sigma_M^2 with a linear fit. Under eta_eff=eta/(1+sigma_M^2+sigma_B^2)
    collapsed onto the (biased) baseline, s = sigma_M^2/(1+sigma_B^2): a line through the
    origin with slope 1/(1+sigma_B^2). Returns the fit dict."""
    ss = sorted(fits.keys())
    x = np.array([s ** 2 for s in ss])
    y = np.array([fits[s]['s'] for s in ss])
    good = np.isfinite(y)
    slope = intercept = r2 = sigma_B_sq = float('nan')
    if good.sum() >= 2:
        A = np.vstack([x[good], np.ones(good.sum())]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, y[good], rcond=None)
        ss_res = np.sum((y[good] - (slope * x[good] + intercept)) ** 2)
        ss_tot = np.sum((y[good] - y[good].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        sigma_B_sq = float(1.0 / slope - 1.0) if slope > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in ss:
        ax.scatter([s ** 2], [fits[s]['s']], s=48, color=colors[s], zorder=3,
                   label=f"$\\sigma_M$={s:.3f}")
    xline = np.linspace(0, float(x.max()) if x.size else 1, 100)
    if np.isfinite(slope):
        ax.plot(xline, slope * xline + intercept, color='crimson', lw=1.8,
                label=f'fit: slope={slope:.2f}, int={intercept:.2f}, $R^2$={r2:.3f}')
    ax.set_xlabel(r'$\sigma_M^2$')
    ax.set_ylabel(r'fitted $s$')
    ax.set_title('constant $s$ vs batch strength (pbmc)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2),
                sigma_B_sq_from_slope=sigma_B_sq)


def generate(results_dir, mean_umi_full=None):
    """Fit constant-s per curve, build figures + summary_const_s.json from results.csv."""
    agg, meta = load(results_dir)
    agg, mean_umi_full, eta_note = resolve_eta(agg, results_dir, mean_umi_full)
    colors = sigma_M_palette(sorted(agg['sigma_M'].unique()))
    eta_b, mi_b = baseline_curve(agg)

    s_grid = np.linspace(0.0, 60.0, 6001)
    fits = {}
    for s in sorted(agg['sigma_M'].unique()):
        if np.isclose(s, 0.0):
            continue
        eta, mi = curve(agg, s)
        fits[s] = fit_s_curve(eta, mi, eta_b, mi_b, s_grid)

    fig1_raw(agg, colors, os.path.join(results_dir, 'fig1_raw_curves.png'))
    fig2_collapse(agg, colors, fits, os.path.join(results_dir, 'fig2_collapse_const_s.png'))
    lin = fig3_s_vs_sigmaM2(fits, colors, os.path.join(results_dir, 'fig3_s_vs_sigmaM2.png'))

    summary = dict(
        model='eta_eff = eta / (1 + s), s constant per sigma_M curve; eta = mean UMI/cell',
        eta_mean_umi_full=mean_umi_full,
        eta_note=eta_note,
        s_per_curve={f"{s:.6f}": fits[s] for s in sorted(fits.keys())},
        s_vs_sigmaM2_linfit=lin,
        dataset=meta.get('dataset'),
        classifier=meta.get('classifier'),
        mi_estimator=meta.get('mi_estimator'),
        k=meta.get('k'),
        n_train=meta.get('n_train'),
        n_test=meta.get('n_test'),
        n_per_batch=(meta.get('n_train') / meta.get('k')) if meta.get('n_train') and meta.get('k') else None,
        epochs=meta.get('epochs'),
        seeds=meta.get('seeds'),
        noise_grid=meta.get('noise_grid'),
        sigma_M_grid=meta.get('sigma_M_grid'),
    )
    with open(os.path.join(results_dir, 'summary_const_s.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description="PBMC constant-s (eta/(1+s)) collapse analysis.")
    p.add_argument('--results-dir', default=str(Path(__file__).resolve().parent / 'results_pbmc'))
    p.add_argument('--mean-umi-full', type=float, default=None,
                   help='full-depth mean UMI/cell over HVGs (overrides the cached value)')
    args = p.parse_args()
    summary = generate(args.results_dir, args.mean_umi_full)
    print('wrote fig1_raw_curves.png, fig2_collapse_const_s.png, '
          'fig3_s_vs_sigmaM2.png, summary_const_s.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
