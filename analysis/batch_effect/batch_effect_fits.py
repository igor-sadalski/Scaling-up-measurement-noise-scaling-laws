"""batch_effect_fits.py

Unified fit + figure for the image (tissuemnist) and PBMC (CITE-seq) batch-effect sweeps.

Model (both datasets, one alpha fit PER sigma_M curve):

    eta_eff = eta / (1 + alpha * eta * sigma_M^2)

Each non-baseline curve is rescaled onto the sigma_M=0 baseline; alpha_curve minimizes the
squared residual of MI vs the baseline MI interpolated at eta_eff (log-eta space).

Figure: 2x2 panels. Top row = raw MI curves, bottom row = collapsed onto eta_eff.
Left column = image, right column = PBMC. Style follows analysis/rebuttal_figures.

Run with the `lt` env:  python analysis/batch_effect/batch_effect_fits.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib import rcParams
import matplotlib as mpl
from scipy.optimize import minimize_scalar
from lmfit import Model

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Number of baseline (sigma_M=0) points to hide, from the lowest eta up, per dataset.
# Display only: dropped points are removed from the drawn curves; the fit still uses the
# full measured baseline.
DROP_BASELINE = {"image": 0, "pbmc": 3}

# ---- style (matches analysis/rebuttal_figures/utils.py) ---------------------
sns.set_style("whitegrid")
rcParams.update({
    "figure.dpi": 150,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "grid.linewidth": 0.5,
    "grid.linestyle": "dashed",
    "legend.fancybox": False,
    "mathtext.fontset": "stix",
})
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


# ---- data loading -----------------------------------------------------------
def _agg(results_dir):
    """Mean + SEM over seeds per (eta axis, sigma_M) cell."""
    df = pd.read_csv(os.path.join(results_dir, "results.csv"))
    g = (df.groupby(["noise_level", "sigma_M"], as_index=False)
           .agg(MI=("MI", "mean"),
                MI_sem=("MI", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
                eta=("eta", "first"),
                n_seeds=("MI", "size")))
    return g


def load_image():
    """eta = 1/sigma_meas^2 (already in the eta column)."""
    return _agg(os.path.join(HERE, "results_image"))


def load_pbmc():
    """eta = mean UMI/cell = quality * full-depth mean UMI/cell over HVGs."""
    rdir = os.path.join(HERE, "results_pbmc")
    g = _agg(rdir)
    mean_umi_full = json.load(open(os.path.join(rdir, "mean_umi_full.json")))["mean_umi_per_cell_hvg"]
    g = g.copy()
    g["eta"] = g["eta"] * mean_umi_full   # eta column held quality; scale to mean UMI/cell
    return g


# ---- fit --------------------------------------------------------------------
def drop_low_baseline(agg, n):
    """Drop the n lowest-eta baseline (sigma_M=0) points; non-baseline curves untouched."""
    if n <= 0:
        return agg
    base = agg[np.isclose(agg["sigma_M"], 0.0)].sort_values("eta")
    return agg.drop(index=base.index[:n]).reset_index(drop=True)


def baseline_curve(agg):
    """sigma_M == 0 curve, sorted by eta ascending."""
    b = agg[np.isclose(agg["sigma_M"], 0.0)].sort_values("eta")
    return b["eta"].to_numpy(), b["MI"].to_numpy()


def baseline_MI_at_eta(eta_query, eta_b, mi_b):
    """Baseline MI interpolated at eta in log-eta space (clamped past the measured range)."""
    order = np.argsort(eta_b)
    return np.interp(np.log(np.atleast_1d(eta_query)), np.log(eta_b[order]), mi_b[order])


def fit_alpha(agg):
    """Fit alpha per sigma_M curve for eta_eff = eta / (1 + alpha*eta*sigma_M^2), minimizing the
    squared residual of MI vs the baseline interpolated at eta_eff. alpha >= 0; optimized in
    log space. Returns {sigma_M: dict(alpha, residual_mse, n_points)}."""
    eta_b, mi_b = baseline_curve(agg)
    fits = {}
    for s, sub in _nonbaseline_curves(agg):
        def mse(alpha):
            return float(np.mean(_curve_resid(sub, s, alpha, eta_b, mi_b) ** 2))
        alpha = _minimize_alpha(mse)
        fits[s] = dict(alpha=float(alpha), residual_mse=mse(alpha), n_points=int(len(sub)))
    return fits


def fit_shared_alpha(agg):
    """Fit a SINGLE alpha shared across all non-baseline curves, minimizing the pooled squared
    residual of every point vs the baseline. Returns (alpha, {sigma_M: dict(alpha, residual_mse,
    n_points)}) where each curve's dict carries the shared alpha and its own residual at it."""
    eta_b, mi_b = baseline_curve(agg)
    curves = _nonbaseline_curves(agg)

    def mse(alpha):
        res = np.concatenate([_curve_resid(sub, s, alpha, eta_b, mi_b) for s, sub in curves])
        return float(np.mean(res ** 2))

    alpha = _minimize_alpha(mse)
    fits = {s: dict(alpha=float(alpha),
                    residual_mse=float(np.mean(_curve_resid(sub, s, alpha, eta_b, mi_b) ** 2)),
                    n_points=int(len(sub)))
            for s, sub in curves}
    return float(alpha), fits


def _nonbaseline_curves(agg):
    """[(sigma_M, sub-frame sorted by eta)] for every sigma_M > 0."""
    return [(float(s), agg[np.isclose(agg["sigma_M"], s)].sort_values("eta"))
            for s in sorted(agg["sigma_M"].unique()) if not np.isclose(s, 0.0)]


def _curve_resid(sub, s, alpha, eta_b, mi_b):
    """MI - baseline_MI_at(eta_eff) for one curve at a given alpha."""
    eta, mi = sub["eta"].to_numpy(), sub["MI"].to_numpy()
    eta_eff = eta / (1.0 + alpha * eta * s ** 2)
    return mi - baseline_MI_at_eta(eta_eff, eta_b, mi_b)


def _minimize_alpha(mse):
    """alpha >= 0 minimizing mse(alpha), searched in log space with alpha=0 as a floor."""
    res = minimize_scalar(lambda t: mse(10.0 ** t), bounds=(-10.0, 6.0), method="bounded")
    alpha = 10.0 ** float(res.x)
    return 0.0 if mse(0.0) <= mse(alpha) else alpha


def info_scaling(eta, eta_bar, I_max):
    """Eq. 1: I(eta) = I_max - 0.5*log2((eta/eta_bar + 1)/(eta/eta_bar + 2^(-2 I_max)))."""
    eps = 1e-12
    r = np.asarray(eta) / (eta_bar if eta_bar != 0 else eps)
    denom = np.where(r + 2 ** (-2 * I_max) == 0, eps, r + 2 ** (-2 * I_max))
    ratio = np.where((r + 1) / denom <= 0, eps, (r + 1) / denom)
    return I_max - 0.5 * np.log2(ratio)


def fit_info_scaling(eta, mi):
    """lmfit fit of Eq. 1 to (eta, mi); mirrors the noise-scaling fit in big_fig_2.py."""
    model = Model(info_scaling)
    params = model.make_params(eta_bar=dict(value=float(np.median(eta)), min=0),
                               I_max=dict(value=float(np.max(mi)), min=0))
    return model.fit(np.asarray(mi, float), params, eta=np.asarray(eta, float))


# ---- plotting ---------------------------------------------------------------
BASELINE_COLOR = "0.6"   # light gray


def sigma_M_palette(sigma_vals):
    cmap = plt.get_cmap("viridis")
    nz = [s for s in sigma_vals if not np.isclose(s, 0.0)]
    return {s: cmap(i / max(1, len(nz) - 1)) for i, s in enumerate(nz)}


def _plot_raw(ax, agg, colors, xlabel):
    for s in sorted(agg["sigma_M"].unique()):
        sub = agg[np.isclose(agg["sigma_M"], s)].sort_values("eta")
        is_base = np.isclose(s, 0.0)
        ax.errorbar(sub["eta"], sub["MI"], yerr=sub["MI_sem"],
                    marker="o", ms=3.5, lw=2.2 if is_base else 1.4, capsize=2, alpha=0.7,
                    zorder=1 if is_base else 3,
                    color=BASELINE_COLOR if is_base else colors[s],
                    label="baseline" if is_base else rf"$\sigma_M$={s:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MI (bits)")


def _plot_collapse(ax, agg, colors, fits, xlabel, per_curve_alpha=True):
    """Scatter the collapsed points and overlay a dotted Eq. 1 fit to all of them."""
    eta_b, mi_b = baseline_curve(agg)
    ax.errorbar(eta_b, mi_b, marker="o", ms=3.5, ls="none", color=BASELINE_COLOR,
                alpha=0.7, zorder=2, label="baseline")
    eta_all, mi_all = [eta_b], [mi_b]
    for s, sub in _nonbaseline_curves(agg):
        alpha = fits[s]["alpha"]
        eta_eff = sub["eta"].to_numpy() / (1.0 + alpha * sub["eta"].to_numpy() * s ** 2)
        mi = sub["MI"].to_numpy()
        label = (rf"$\sigma_M$={s:.2f} ($\alpha$={alpha:.2g})" if per_curve_alpha
                 else rf"$\sigma_M$={s:.2f}")
        ax.errorbar(eta_eff, mi, yerr=sub["MI_sem"], marker="s", ms=3.5, ls="none",
                    capsize=2, alpha=0.7, zorder=3, color=colors[s], label=label)
        eta_all.append(eta_eff); mi_all.append(mi)

    eta_all, mi_all = np.concatenate(eta_all), np.concatenate(mi_all)
    result = fit_info_scaling(eta_all, mi_all)
    xf = np.logspace(np.log10(eta_all.min()), np.log10(eta_all.max()), 400)
    ax.plot(xf, result.eval(eta=xf), ls=":", color="k", lw=1.4, zorder=4, label="Eq. 1 fit")

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MI (bits)")
    return dict(eta_bar=float(result.params["eta_bar"].value),
                I_max=float(result.params["I_max"].value),
                r2=float(result.rsquared))


ETA_IMG = r"$\eta = \mathrm{noise\ variance}^{-1}$"
ETA_PBMC = r"$\eta$ = mean UMI/cell"
ETA_EFF = r"$\eta_{\mathrm{eff}} = \eta/(1+\alpha\,\eta\,\sigma_M^2)$"


def make_figure(shared=False, name="batch_effect_collapse"):
    """Build the 2x2 figure. shared=False fits one alpha per curve; shared=True fits a single
    alpha across all curves (per dataset), annotated in the collapse panels."""
    img_full, pbmc_full = load_image(), load_pbmc()
    if shared:
        img_alpha, img_fits = fit_shared_alpha(img_full)
        pbmc_alpha, pbmc_fits = fit_shared_alpha(pbmc_full)
    else:
        img_fits, pbmc_fits = fit_alpha(img_full), fit_alpha(pbmc_full)
        img_alpha = pbmc_alpha = None
    # display copies with the lowest-eta baseline points hidden (fit used the full baseline above)
    img = drop_low_baseline(img_full, DROP_BASELINE["image"])
    pbmc = drop_low_baseline(pbmc_full, DROP_BASELINE["pbmc"])
    img_colors = sigma_M_palette(sorted(img["sigma_M"].unique()))
    pbmc_colors = sigma_M_palette(sorted(pbmc["sigma_M"].unique()))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    (ax_ir, ax_pr), (ax_ic, ax_pc) = axes

    _plot_raw(ax_ir, img, img_colors, ETA_IMG)
    _plot_raw(ax_pr, pbmc, pbmc_colors, ETA_PBMC)
    img_scaling = _plot_collapse(ax_ic, img, img_colors, img_fits, ETA_EFF, per_curve_alpha=not shared)
    pbmc_scaling = _plot_collapse(ax_pc, pbmc, pbmc_colors, pbmc_fits, ETA_EFF, per_curve_alpha=not shared)

    if shared:
        ax_ic.text(0.04, 0.96, rf"shared $\alpha$={img_alpha:.2g}", transform=ax_ic.transAxes,
                   va="top", ha="left", fontsize=7)
        ax_pc.text(0.04, 0.96, rf"shared $\alpha$={pbmc_alpha:.2g}", transform=ax_pc.transAxes,
                   va="top", ha="left", fontsize=7)

    ax_ir.set_title("TissueMNIST (images)")
    ax_pr.set_title("PBMC CITE-seq")
    for ax in (ax_ir, ax_pr, ax_ic, ax_pc):
        ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = dict(
        model=("eta_eff = eta / (1 + alpha * eta * sigma_M^2), single alpha shared across curves"
               if shared else
               "eta_eff = eta / (1 + alpha * eta * sigma_M^2), alpha fit per curve"),
        image={f"{s:.4f}": img_fits[s] for s in sorted(img_fits)},
        pbmc={f"{s:.4f}": pbmc_fits[s] for s in sorted(pbmc_fits)},
        eq1_scaling_fit=dict(
            model="I = I_max - 0.5*log2((eta/eta_bar + 1)/(eta/eta_bar + 2^(-2 I_max)))",
            image=img_scaling, pbmc=pbmc_scaling),
    )
    if shared:
        summary["image_shared_alpha"] = img_alpha
        summary["pbmc_shared_alpha"] = pbmc_alpha
    with open(os.path.join(FIGDIR, f"{name.replace('collapse', 'fits')}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    per_curve = make_figure(shared=False, name="batch_effect_collapse")
    shared = make_figure(shared=True, name="batch_effect_collapse_shared_alpha")
    print("wrote figures/batch_effect_collapse{,_shared_alpha}.{png,pdf} and matching fits json")
    print("shared alpha -> image:", shared["image_shared_alpha"], "pbmc:", shared["pbmc_shared_alpha"])
