# analyze_batch_effect.py
# Reads results/results.csv and writes fig1 (raw curves), fig2 (collapse, two
# variants), fig3 (empirical map), and summary.json.
# Theory: eta_eff = eta / (1 + sigma_M^2/sigma_meas^2), eta = 1/sigma_meas^2.

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
                  eta=('eta', 'first'),
                  sigma_meas_sq=('sigma_meas_sq', 'first'),
                  sigma_M_sq=('sigma_M_sq', 'first'),
                  ratio=('sigma_M_sq_ratio', 'first'),
                  n_seeds=('MI', 'size')))
    meta_path = os.path.join(results_dir, 'run_meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return agg, meta


def baseline_curve(agg):
    """sigma_M == 0 curve, sorted by eta ascending."""
    b = agg[np.isclose(agg['sigma_M'], 0.0)].sort_values('eta')
    return b['eta'].to_numpy(), b['MI'].to_numpy()


def baseline_MI_at_eta(eta_query, eta_b, mi_b):
    """Baseline MI interpolated at eta (log-eta space)."""
    order = np.argsort(eta_b)
    return np.interp(np.log(eta_query), np.log(eta_b[order]), mi_b[order])


def baseline_eta_at_MI(mi_query, eta_b, mi_b):
    """Baseline eta yielding a given MI; NaN outside the baseline MI range."""
    order = np.argsort(mi_b)
    mi_sorted, eta_sorted = mi_b[order], eta_b[order]
    out = np.interp(mi_query, mi_sorted, eta_sorted)
    return np.where((mi_query < mi_sorted[0]) | (mi_query > mi_sorted[-1]), np.nan, out)


def sigma_M_palette(sigma_vals):
    cmap = plt.get_cmap('viridis')
    return {s: cmap(i / max(1, len(sigma_vals) - 1)) for i, s in enumerate(sigma_vals)}


def collapse_mse(agg, eta_b, mi_b, alpha):
    """Mean squared residual of alpha-rescaled curves vs the baseline (in-range points)."""
    res = []
    for s in sorted(agg['sigma_M'].unique()):
        if np.isclose(s, 0.0):
            continue
        sub = agg[np.isclose(agg['sigma_M'], s)].sort_values('eta')
        eta_eff = sub['eta'].to_numpy() / (1.0 + alpha * sub['ratio'].to_numpy())
        mi = sub['MI'].to_numpy()
        inr = (eta_eff >= eta_b.min()) & (eta_eff <= eta_b.max())
        if inr.any():
            res.append((mi[inr] - baseline_MI_at_eta(eta_eff[inr], eta_b, mi_b)) ** 2)
    return float(np.mean(np.concatenate(res))) if res else float('nan')


def fit_alpha(agg, eta_b, mi_b, lo=0.0, hi=8.0, n=801):
    """alpha minimizing the collapse residual."""
    grid = np.linspace(lo, hi, n)
    mses = [collapse_mse(agg, eta_b, mi_b, a) for a in grid]
    return float(grid[int(np.nanargmin(mses))])


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
    ax.set_xlabel(r'$\eta = 1/\sigma_{\mathrm{meas}}^2$')
    ax.set_ylabel('mi (bits)')
    ax.set_title('raw mi curves')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_collapse(agg, colors, out, alpha=1.0, title='collapse', xlabel=None):
    """MI vs alpha-rescaled eta_eff; overlays the baseline. Returns residual MSE per sigma_M."""
    eta_b, mi_b = baseline_curve(agg)
    resid_mse = {}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(eta_b, mi_b, color='k', lw=2.4, marker='o', ms=4, zorder=5, label='baseline')
    for s in sorted(agg['sigma_M'].unique()):
        if np.isclose(s, 0.0):
            continue
        sub = agg[np.isclose(agg['sigma_M'], s)].sort_values('eta')
        eta_eff = sub['eta'].to_numpy() / (1.0 + alpha * sub['ratio'].to_numpy())
        mi = sub['MI'].to_numpy()
        ax.plot(eta_eff, mi, marker='s', ms=4, lw=1.4, color=colors[s],
                label=f"$\\sigma_M$={s:.3f}")
        inr = (eta_eff >= eta_b.min()) & (eta_eff <= eta_b.max())
        resid_mse[f"{s:.6f}"] = (float(np.mean((mi[inr] - baseline_MI_at_eta(eta_eff[inr], eta_b, mi_b)) ** 2))
                                 if inr.any() else float('nan'))
    ax.set_xscale('log')
    ax.set_xlabel(xlabel or r'$\eta_{\mathrm{eff}} = \eta/(1+\sigma_M^2/\sigma_{\mathrm{meas}}^2)$')
    ax.set_ylabel('mi (bits)')
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return resid_mse


def fig3_empirical(agg, colors, out):
    eta_b, mi_b = baseline_curve(agg)
    xs, ys, cs = [], [], []
    for s in sorted(agg['sigma_M'].unique()):
        if np.isclose(s, 0.0):
            continue
        sub = agg[np.isclose(agg['sigma_M'], s)].sort_values('eta')
        eta, mi, ratio = sub['eta'].to_numpy(), sub['MI'].to_numpy(), sub['ratio'].to_numpy()
        eta_eff_fit = baseline_eta_at_MI(mi, eta_b, mi_b)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = eta / eta_eff_fit
        good = np.isfinite(y) & (y > 0)
        xs.append(ratio[good]); ys.append(y[good]); cs.append(np.full(good.sum(), s))

    X = np.concatenate(xs) if xs else np.array([])
    Y = np.concatenate(ys) if ys else np.array([])
    C = np.concatenate(cs) if cs else np.array([])

    slope = intercept = r2 = float('nan')
    if len(X) >= 2:
        A = np.vstack([X, np.ones_like(X)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, Y, rcond=None)
        ss_res = np.sum((Y - (slope * X + intercept)) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in sorted(set(C.tolist())):
        m = np.isclose(C, s)
        ax.scatter(X[m], Y[m], s=36, color=colors[s], label=f"$\\sigma_M$={s:.3f}", zorder=3)
    xline = np.linspace(0, float(np.nanmax(X)) if len(X) else 1, 100)
    ax.plot(xline, 1 + xline, 'k--', lw=1.5, label=r'theory $1+\sigma_M^2/\sigma_{\mathrm{meas}}^2$')
    if np.isfinite(slope):
        ax.plot(xline, slope * xline + intercept, color='crimson', lw=1.8,
                label=f'fit: slope={slope:.2f}, int={intercept:.2f}, $R^2$={r2:.3f}')
    ax.set_xlabel(r'$\sigma_M^2/\sigma_{\mathrm{meas}}^2$')
    ax.set_ylabel(r'$\eta/\eta_{\mathrm{eff}}^{\mathrm{fit}}$')
    ax.set_title('empirical mapping')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), n_points=int(len(X)))


def generate(results_dir):
    """Build figures + summary.json from results_dir/results.csv."""
    agg, meta = load(results_dir)
    colors = sigma_M_palette(sorted(agg['sigma_M'].unique()))
    eta_b, mi_b = baseline_curve(agg)

    fig1_raw(agg, colors, os.path.join(results_dir, 'fig1_raw_curves.png'))

    # theory (alpha = 1) and best-fit alpha
    resid_theory = fig2_collapse(
        agg, colors, os.path.join(results_dir, 'fig2_collapse.png'),
        alpha=1.0, title='collapse (theory)')
    alpha = fit_alpha(agg, eta_b, mi_b)
    resid_alpha = fig2_collapse(
        agg, colors, os.path.join(results_dir, 'fig2_collapse_alpha.png'),
        alpha=alpha, title=f'collapse (fit $\\alpha$={alpha:.2f})',
        xlabel=r'$\eta_{\mathrm{eff}} = \eta/(1+\alpha\,\sigma_M^2/\sigma_{\mathrm{meas}}^2)$')

    fit = fig3_empirical(agg, colors, os.path.join(results_dir, 'fig3_empirical_map.png'))

    summary = dict(
        theory='eta_eff = eta / (1 + alpha * sigma_M^2/sigma_meas^2), eta = 1/sigma_meas^2',
        alpha_fit=float(alpha),
        collapse_residual_mse_theory=resid_theory,
        collapse_residual_mse_alpha=resid_alpha,
        empirical_fit=fit,
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
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=str(Path(__file__).resolve().parent / 'results'))
    args = p.parse_args()
    summary = generate(args.results_dir)
    print('wrote fig1_raw_curves.png, fig2_collapse.png, fig2_collapse_alpha.png, '
          'fig3_empirical_map.png, summary.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
