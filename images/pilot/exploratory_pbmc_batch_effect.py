# exploratory_pbmc_batch_effect.py
# For each (sigma_M, eta) point, invert the measured sigma_M=0 baseline to find the empirical
# eta_eff whose baseline MI equals the observed MI (no functional form assumed). Then look at
# the empirical rescale factor eta/eta_eff vs sigma_M^2, hued by eta, to see the form.
# Reads only precomputed results_pbmc/results.csv; writes pilot_*.png/csv to the same folder.

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
    agg = (df.groupby(['sigma_M', 'quality'], as_index=False)
             .agg(MI=('MI', 'mean'), sigma_M_sq=('sigma_M_sq', 'first')))
    cache = os.path.join(results_dir, 'mean_umi_full.json')
    mean_umi_full = float(json.load(open(cache))['mean_umi_per_cell_hvg']) if os.path.exists(cache) else 1.0
    agg['eta'] = agg['quality'] * mean_umi_full
    return agg


def invert_baseline(mi_query, eta_b, mi_b):
    """eta_eff such that baseline_MI(eta_eff) = mi_query, via log-eta interp on the measured
    baseline. np.interp clamps outside the measured MI range."""
    o = np.argsort(mi_b)
    return np.exp(np.interp(mi_query, mi_b[o], np.log(eta_b)[o]))


def baseline_mi_at(eta_query, eta_b, mi_b):
    o = np.argsort(eta_b)
    return np.interp(np.log(eta_query), np.log(eta_b)[o], mi_b[o])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=str(Path(__file__).resolve().parent / 'results_pbmc'))
    args = p.parse_args()
    d = args.results_dir

    agg = load(d)
    base = agg[np.isclose(agg['sigma_M'], 0.0)].sort_values('eta')
    eta_b, mi_b = base['eta'].to_numpy(), base['MI'].to_numpy()

    pts = agg[~np.isclose(agg['sigma_M'], 0.0)].copy()
    pts['eta_eff'] = invert_baseline(pts['MI'].to_numpy(), eta_b, mi_b)
    pts['ratio'] = pts['eta'] / pts['eta_eff']
    pts['clamped'] = (pts['MI'] < mi_b.min()) | (pts['MI'] > mi_b.max())
    pts = pts.sort_values(['eta', 'sigma_M'])

    etas = sorted(pts['eta'].unique())
    cmap = plt.get_cmap('viridis')
    colors = {e: cmap(i / max(1, len(etas) - 1)) for i, e in enumerate(etas)}

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for e in etas:
        sub = pts[pts['eta'] == e].sort_values('sigma_M_sq')
        ax.plot(sub['sigma_M_sq'], sub['ratio'], marker='o', ms=5, lw=1.4,
                color=colors[e], label=f'$\\eta$={e:.1f}')
        clamp = sub[sub['clamped']]
        ax.scatter(clamp['sigma_M_sq'], clamp['ratio'], s=90, facecolors='none',
                   edgecolors='crimson', zorder=5)
    ax.axhline(1.0, color='0.6', lw=0.8, ls='--')
    ax.set_xlabel(r'$\sigma_M^2$')
    ax.set_ylabel(r'$\eta/\eta_{\mathrm{eff}}$  (empirical rescale factor)')
    ax.set_title('empirical rescale factor vs batch strength (pbmc)\n'
                 'red rings = MI outside measured baseline range')
    ax.legend(fontsize=8, frameon=False, title='mean UMI/cell', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'pilot_empirical_eta_eff.png'), dpi=150)
    plt.close(fig)

    # collapse check: if eta/eta_eff = 1 + alpha*eta*sigma_M^2 then (ratio-1)/eta vs sigma_M^2
    # is a single line (slope alpha) independent of eta -- the curves should overlap.
    pts['collapse'] = (pts['ratio'] - 1.0) / pts['eta']
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for e in etas:
        sub = pts[pts['eta'] == e].sort_values('sigma_M_sq')
        ax.plot(sub['sigma_M_sq'], sub['collapse'], marker='o', ms=5, lw=1.4,
                color=colors[e], label=f'$\\eta$={e:.1f}')
    ax.set_xlabel(r'$\sigma_M^2$')
    ax.set_ylabel(r'$(\eta/\eta_{\mathrm{eff}} - 1)\,/\,\eta$')
    ax.set_title('collapse check: form $\\eta/\\eta_{\\mathrm{eff}} = 1 + \\alpha\\,\\eta\\,\\sigma_M^2$\n'
                 'curves overlap iff the form holds (slope = $\\alpha$)')
    ax.legend(fontsize=8, frameon=False, title='mean UMI/cell', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'pilot_collapse_check.png'), dpi=150)
    plt.close(fig)

    # collapse the real curves with eta_eff = eta / (1 + alpha*eta*sigma_M^2). Fit one global
    # alpha minimizing MSE of every non-baseline point against the baseline MI at its eta_eff.
    e_arr, s_arr, mi_arr = pts['eta'].to_numpy(), pts['sigma_M_sq'].to_numpy(), pts['MI'].to_numpy()
    a_grid = np.linspace(0.0, 1.0, 20001)
    mse = [np.mean((mi_arr - baseline_mi_at(e_arr / (1 + a * e_arr * s_arr), eta_b, mi_b)) ** 2)
           for a in a_grid]
    alpha = float(a_grid[int(np.argmin(mse))])
    pts['eta_eff_model'] = pts['eta'] / (1 + alpha * pts['eta'] * pts['sigma_M_sq'])

    sigs = sorted(pts['sigma_M'].unique())
    scolors = {s: cmap(i / max(1, len(sigs) - 1)) for i, s in enumerate(sigs)}
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(eta_b, mi_b, color='k', lw=2.4, marker='o', ms=4, zorder=5, label='baseline')
    for s in sigs:
        sub = pts[np.isclose(pts['sigma_M'], s)].sort_values('eta_eff_model')
        ax.plot(sub['eta_eff_model'], sub['MI'], marker='s', ms=5, lw=1.4,
                color=scolors[s], label=f'$\\sigma_M$={s:.2f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\eta_{\mathrm{eff}} = \eta/(1+\alpha\,\eta\,\sigma_M^2)$  (mean UMI/cell)')
    ax.set_ylabel('mi (bits)')
    ax.set_title(f'collapse onto baseline, $\\alpha$={alpha:.4f} (pbmc)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'pilot_collapse_alpha_eta.png'), dpi=150)
    plt.close(fig)

    pts.to_csv(os.path.join(d, 'pilot_empirical_eta_eff.csv'), index=False)
    print(f'fitted alpha = {alpha:.4f}')
    print('wrote pilot_empirical_eta_eff.png, pilot_collapse_check.png, '
          'pilot_collapse_alpha_eta.png, pilot_empirical_eta_eff.csv')
    print(pts[['sigma_M', 'eta', 'MI', 'eta_eff', 'ratio', 'clamped']].to_string(index=False))


if __name__ == '__main__':
    main()
