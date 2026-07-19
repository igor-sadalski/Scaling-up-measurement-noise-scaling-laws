"""Extrapolation test for the noise-scaling law (Reviewer 3, Major Comment 3.4).

Transcriptomic curves are noise-limited, so their saturation value I_max is
extrapolated rather than observed. Here we use image (TissueMNIST) and protein
sequence (spike) curves, which *are* measured beyond saturation, to ask when the
fitted law can recover I_max from the unsaturated part of a curve alone.

For each curve we fit the law to the lowest-SNR points only, progressively
including more of the curve, and compare the extrapolated I_max to the observed
saturation plateau. Extrapolation is accurate once the fit reaches the onset of
saturation, and under-determined when only the linear regime is observed.

Panel (a): a representative curve with the fit obtained from its noise-limited
points extrapolated through the withheld (saturated) points.
Panel (b): extrapolation accuracy across all curves and truncation depths.

Writes figures/08_extrapolation_test.png and figures/08_extrapolation_test.csv.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

REPO_ROOT = os.path.join(utils.HERE, "..", "..")
SEQ_CSV = os.path.join(REPO_ROOT, "seq", "multisize_gisaid_results.csv")
TISSUE_GLOB = os.path.join(REPO_ROOT, "images", "tissuemnist_models", "result_*.csv")

# Palette distinct from the method/metric colours used elsewhere in the paper:
# a green ramp for the three sequence models, a magenta pair for the two image
# corruptions. Domain is also encoded by marker shape (circle vs square).
CURVE_STYLE = {
    "Spike seq 8M":       dict(color="#74c476", marker="o"),
    "Spike seq 35M":      dict(color="#238b45", marker="o"),
    "Spike seq 150M":     dict(color="#00441b", marker="o"),
    "TissueMNIST (blur)": dict(color="#f768a1", marker="s"),
    "TissueMNIST (pixel)":dict(color="#ae017e", marker="s"),
}


def imax_from_fit(result):
    A, B = result.params["A"].value, result.params["B"].value
    if A <= 0 or B <= 0:
        return np.nan
    return 0.5 * np.log2(B / A)


def load_curves():
    """Return {name: (x_snr, y_mi)} for every saturating curve."""
    curves = {}

    seq = pd.read_csv(SEQ_CSV)
    for size in ["8M", "35M", "150M"]:
        s = seq[seq["model_size"] == size].sort_values("true/error")
        curves[f"Spike seq {size}"] = (s["true/error"].values, s["mutual_information"].values)

    rows = []
    for f in glob.glob(TISSUE_GLOB):
        tok = re.search(r"result_(.+)\.csv", f).group(1)
        if "gauss" in tok:
            typ, lvl = "blur", float(tok.split("_")[1])
        elif "pix" in tok:
            typ, lvl = "pixel", float(re.sub(r"[^0-9.]", "", tok.split("_")[1]))
        else:
            continue
        rows.append((typ, lvl, pd.read_csv(f)["mi_score"].iloc[0]))
    tissue = pd.DataFrame(rows, columns=["type", "level", "mi"])
    for typ, label in [("blur", "TissueMNIST (blur)"), ("pixel", "TissueMNIST (pixel)")]:
        s = tissue[tissue["type"] == typ].copy()
        s["x"] = 1.0 / s["level"]
        if typ == "pixel":
            s["x"] = s["x"] ** 2
        s = s.sort_values("x")
        curves[label] = (s["x"].values, s["mi"].values)
    return curves


def plateau_of(y):
    """Observed saturation: mean of points within 5% of the maximum."""
    return float(np.mean(y[y >= 0.95 * y.max()]))


def sweep(x, y):
    """Fit the lowest-j SNR points for j = 3..n; return arrays of
    (normalized SNR range measured, extrapolated I_max)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    plateau = plateau_of(y)
    lo, hi = np.log10(x.min()), np.log10(x.max())
    snr_norm, imax = [], []
    for j in range(3, len(x) + 1):
        r = utils.fit_noise(x[:j], y[:j])
        snr_norm.append((np.log10(x[:j].max()) - lo) / (hi - lo))
        imax.append(imax_from_fit(r))
    return np.array(snr_norm), np.array(imax), plateau


def main():
    curves = load_curves()

    # ---- panel (a): representative curve, fit from noise-limited points ----
    rep = "Spike seq 8M"
    x, y = curves[rep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    plateau = plateau_of(y)
    n_withheld = 3  # withhold the most-saturated points
    used = slice(0, len(x) - n_withheld)
    res = utils.fit_noise(x[used], y[used])
    imax_ex = imax_from_fit(res)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    style = CURVE_STYLE[rep]
    xf = np.logspace(np.log10(x.min() / 2), np.log10(x.max() * 2), 400)
    axa.plot(xf, res.eval(x=xf), "--", color=style["color"], lw=1.4, zorder=1)
    axa.axhline(plateau, ls=":", color="0.4", lw=1)
    axa.scatter(x[used], y[used], color=style["color"], s=26, zorder=3,
                label="fit to noise-limited points")
    axa.scatter(x[len(x) - n_withheld:], y[len(x) - n_withheld:], s=26, zorder=3,
                facecolors="none", edgecolors=style["color"], linewidths=1.2,
                label="withheld (saturated)")
    axa.set_xscale("log")
    axa.set_xlabel("signal-to-noise (true/error)")
    axa.set_ylabel("MI (bits)")
    axa.set_title(rep, fontsize=8)
    axa.legend(loc="lower right", fontsize=6)
    axa.text(0.04, 0.94,
             f"observed $I_{{max}}$ = {plateau:.2f}\nextrapolated = {imax_ex:.2f}",
             transform=axa.transAxes, va="top", ha="left", fontsize=6.5)

    # ---- panel (b): accuracy across all curves and truncation depths ----
    records = []
    axb.axhline(0.0, color="0.4", lw=1, ls=":", zorder=1)
    for name, (cx, cy) in curves.items():
        snr_norm, imax, plat = sweep(cx, cy)
        err = imax - plat
        st = CURVE_STYLE[name]
        axb.scatter(snr_norm, err, color=st["color"], marker=st["marker"], s=32,
                    edgecolors="white", linewidths=0.5, zorder=3, label=name)
        for sn, im, e in zip(snr_norm, imax, err):
            records.append(dict(curve=name, plateau=plat, snr_range_norm=sn,
                                extrap_Imax=im, error_bits=e))

    axb.set_xlim(0, 1.02)
    axb.set_xlabel("normalized SNR range measured")
    axb.set_ylabel(r"$I_{max}$ (extrap.) $-$ $I_{max}$ (obs.)  (bits)")
    axb.legend(loc="upper right", fontsize=5.5)

    fig.tight_layout()
    utils.save(fig, "08_extrapolation_test")

    out = os.path.join(utils.FIGDIR, "08_extrapolation_test.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"wrote {out}")
    # Headline numbers: withhold the 2 most-saturated points per curve.
    print("\nExtrapolation withholding the 2 most-saturated points:")
    for name, (cx, cy) in curves.items():
        order = np.argsort(cx)
        cx, cy = cx[order], cy[order]
        plat = plateau_of(cy)
        r = utils.fit_noise(cx[:-2], cy[:-2])
        print(f"  {name:22s} observed={plat:.2f}  extrapolated={imax_from_fit(r):.2f}")


if __name__ == "__main__":
    main()
