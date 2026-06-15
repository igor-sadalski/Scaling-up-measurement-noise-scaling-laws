"""big_fig_2.py

Script form of analysis/big_fig_2.ipynb, extended to include the STATE models.

What this does
--------------
1. Loads the single-cell MI results (`collect_mi_results.csv`), the precomputed
   cell-number / noise scaling fit parameters, and the auxiliary modalities
   (Caltech101, ESM2 spike sequences, TissueMNIST).
2. STATE has raw MI sweeps in `collect_mi_results.csv` but no precomputed scaling
   fits.  We fit them here with the *same* lmfit methodology used to generate the
   other models' parameters (see analysis/archive/2025-11-18_17-10_* and _17-20_*),
   then persist the STATE rows into `cell_scaling.csv` / `noise_scaling.csv` so the
   rest of the analysis picks them up too.
3. Reproduces the full 5-row "big figure 2" and adds STATE everywhere the other
   single-cell models appear, using the lavender colour c6[0] as STATE's colour.

Run with the `scaling` conda env:
    conda activate scaling && python analysis/big_fig_2.py
"""

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")  # headless: we save to disk rather than show
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D
from lmfit import Model

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../analysis
REPO_ROOT = os.path.dirname(HERE)                          # repo root
FINAL = os.path.join(HERE, "final_results")

CALTECH_URL = "https://raw.githubusercontent.com/ggdna/scScaling/main/results/"

COLLECT_CSV = os.path.join(FINAL, "collect_mi_results.csv")
CELL_SCALING_CSV = os.path.join(FINAL, "cell_scaling.csv")
NOISE_SCALING_CSV = os.path.join(FINAL, "noise_scaling.csv")
SEQ_CSV = os.path.join(REPO_ROOT, "seq", "multisize_gisaid_results.csv")
TISSUE_GLOB = os.path.join(REPO_ROOT, "images", "tissuemnist_models", "result_*.csv")

OUT_PDF = os.path.join(HERE, "big_fig_2.pdf")
OUT_PNG = os.path.join(HERE, "big_fig_2.png")

# Label renaming (applied to *values* across all columns, like the notebook).
RENAME_DICT = {
    "celltype.l3": "Cell type MI",
    "protein_counts": "Protein MI",
    "clone": "Clonal MI",
    "author_day": "Temporal MI",
    "ng_idx": "Spatial MI",
    "RandomProjection": "Rand. Proj.",
}

# STATE is only evaluated on each dataset's primary signal.
STATE_DATASET_SIGNAL = {
    "PBMC": "protein_counts",
    "larry": "clone",
    "merfish": "ng_idx",
    "shendure": "author_day",
}


# ----------------------------------------------------------------------------
# Model functions (from the notebook)
# ----------------------------------------------------------------------------
def cell_number_scaling(x, N0, s, I_inf):
    """Cell number scaling: I(x) = max(0, I_inf - (x/N0)^(-s))"""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = x / N0
        scaling_term = np.where(ratio > 0, np.exp(-s * np.log(ratio)), 0.0)
        return np.where(x > 0, np.maximum(0, I_inf - scaling_term), np.nan)


def info_scaling(u, u_bar, I_max):
    """Information scaling: I(u) = I_max - 0.5*log2((1+u/u_bar)/(u/u_bar+2^(-2*I_max)))"""
    epsilon = 1e-12
    u, u_bar, I_max = np.asarray(u), np.asarray(u_bar), np.asarray(I_max)
    u_bar_safe = np.where(u_bar == 0, epsilon, u_bar)
    u_over_u_bar = u / u_bar_safe
    numerator = 1 + u_over_u_bar
    denominator = np.where(u_over_u_bar + 2 ** (-2 * I_max) == 0, epsilon, u_over_u_bar + 2 ** (-2 * I_max))
    ratio = np.where(numerator / denominator <= 0, epsilon, numerator / denominator)
    return I_max - 0.5 * np.log2(ratio)


def info_scaling_model(x, A, B):
    """Parameterized info scaling: 0.5*log2((x*B+1)/(1+A*x))"""
    return 0.5 * np.log2((x * B + 1) / (1 + A * x))


def transform_to_z(x, xbar, imax):
    """Transform to collapsed coordinate z"""
    A = 2 ** (-2 * imax)
    return ((x / xbar) + A) / (1 - A)


# ----------------------------------------------------------------------------
# STATE parameter fitting (mirrors the archived fit scripts)
# ----------------------------------------------------------------------------
def _cell_number_scaling_fit_model(x, N0, s, I_inf):
    """Form used for *fitting* (matches archive/2025-11-18_17-10_hyperparam_fits_cell_scaling.py)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, I_inf - (x / N0) ** (-s), np.nan)


def _info_scaling_fit_model(u, u_bar, I_max):
    """Form used for *fitting* (matches archive/2025-11-18_17-20_hyperparm_fits_noise_scaling.py)."""
    eps = 1e-12
    u = np.asarray(u, dtype=float)
    u_bar_safe = np.where(u_bar == 0, eps, u_bar)
    u_over = u / u_bar_safe
    denom = np.where(u_over + 2 ** (-2 * I_max) == 0, eps, u_over + 2 ** (-2 * I_max))
    ratio = np.where((1 + u_over) / denom <= 0, eps, (1 + u_over) / denom)
    return I_max - 0.5 * np.log2(ratio)


def _safe_stderr(param):
    return param.stderr if param.stderr is not None else np.nan


def fit_state_cell_scaling(collect_df, initial_N0=1e4, initial_s=1.0, initial_I_inf=2.5):
    """Fit cell-number scaling for STATE per (dataset, metric, quality).

    Uses the deep-method bounds (N0 in [10, 1e7], s in [0.01, 10], I_inf in [0.1, 10]),
    the same branch the archived script uses for Geneformer/SCVI/State.
    """
    state = collect_df[collect_df["algorithm"] == "State"]
    model = Model(_cell_number_scaling_fit_model)
    rows = []
    for (dataset, signal, quality), grp in state.groupby(["dataset", "signal", "quality"]):
        if len(grp) < 3:
            continue
        x = grp["size"].values.astype(float)
        y = grp["mi_value"].values.astype(float)
        params = model.make_params(
            N0=dict(value=initial_N0, min=10, max=10 ** 7),
            s=dict(value=initial_s, min=0.01, max=10.0),
            I_inf=dict(value=initial_I_inf, min=0.1, max=10.0),
        )
        try:
            result = model.fit(y, params, x=x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    unc = result.eval_uncertainty(params=result.params, x=x, sigma=2)
                except Exception:
                    unc = np.zeros_like(x)
            rows.append({
                "dataset": dataset, "method": "State", "metric": signal, "quality": quality,
                "N0": result.params["N0"].value, "s": result.params["s"].value,
                "I_inf": result.params["I_inf"].value,
                "N0_error": _safe_stderr(result.params["N0"]),
                "s_error": _safe_stderr(result.params["s"]),
                "I_inf_error": _safe_stderr(result.params["I_inf"]),
                "mean_residual": float(np.mean(np.abs(result.residual))),
                "avg_uncertainty_error": float(np.mean(np.abs(unc))),
            })
        except Exception as e:
            print(f"  [cell] STATE fit failed for {dataset}/{signal}/q={quality:.3f}: {e}")
    out = pd.DataFrame(rows)
    num = ["N0", "s", "I_inf", "N0_error", "s_error", "I_inf_error", "mean_residual", "avg_uncertainty_error"]
    out[num] = out[num].round(3)
    return out


def fit_state_noise_scaling(collect_df, initial_u_bar=540, initial_I_max=3):
    """Fit noise (information) scaling for STATE per (dataset, metric, size)."""
    state = collect_df[collect_df["algorithm"] == "State"]
    model = Model(_info_scaling_fit_model)
    rows = []
    for (dataset, signal, size), grp in state.groupby(["dataset", "signal", "size"]):
        if len(grp) < 3:
            continue
        u = grp["umis_per_cell"].values.astype(float)
        y = grp["mi_value"].values.astype(float)
        params = model.make_params(u_bar=dict(value=initial_u_bar, min=0), I_max=dict(value=initial_I_max, min=0))
        try:
            result = model.fit(y, params, u=u)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    unc = result.eval_uncertainty(params=result.params, u=u, sigma=2)
                except Exception:
                    unc = np.zeros_like(u)
            rows.append({
                "dataset": dataset, "method": "State", "metric": signal, "size": size,
                "fitted_u_bar": result.params["u_bar"].value,
                "fitted_I_max": result.params["I_max"].value,
                "u_bar_error": _safe_stderr(result.params["u_bar"]),
                "I_max_error": _safe_stderr(result.params["I_max"]),
                "avg_uncertainty_error": float(np.mean(np.abs(unc))),
            })
        except Exception as e:
            print(f"  [noise] STATE fit failed for {dataset}/{signal}/size={size}: {e}")
    out = pd.DataFrame(rows)
    num = ["fitted_u_bar", "fitted_I_max", "u_bar_error", "I_max_error", "avg_uncertainty_error"]
    out[num] = out[num].round(3)
    return out


def merge_and_persist(existing_csv, state_rows, key_cols):
    """Drop any prior STATE rows from the CSV, append fresh STATE rows, write back.

    Idempotent: re-running replaces (rather than duplicates) STATE rows.  Returns
    the merged dataframe (raw, un-renamed names) for plotting use.
    """
    existing = pd.read_csv(existing_csv)
    existing = existing[existing["method"] != "State"]
    merged = pd.concat([existing, state_rows[existing.columns]], ignore_index=True)
    merged.to_csv(existing_csv, index=False)
    print(f"  wrote {len(state_rows)} STATE rows -> {os.path.relpath(existing_csv, REPO_ROOT)}")
    return merged


# ----------------------------------------------------------------------------
# Plot helpers (from the notebook)
# ----------------------------------------------------------------------------
def plot_points(ax, x, y, color, label, marker="o"):
    ax.plot(x, y, color=color, marker=marker, alpha=0.7, ms=2.5, label=label, lw=0)


def fit_info_model(x_data, y_data, gate=True):
    """Fit info_scaling_model and return (x_bar, i_max, result).

    When ``gate`` is True (default), the fit is only returned if it is
    trustworthy (A/B standard errors smaller than their values) -- used by the
    collapse / auxiliary-modality panels.  When ``gate`` is False the fit is
    returned regardless of its uncertainty (used for the panel-c example curves,
    which should always be drawn).
    """
    model = Model(info_scaling_model)
    params = model.make_params(A=1e-2, B=1e-2)
    params["A"].min = params["B"].min = 0
    try:
        result = model.fit(y_data, params, x=x_data)
    except Exception:
        return None, None, None
    a, b = result.params["A"], result.params["B"]
    good = a.stderr and b.stderr and a.stderr < a.value and b.stderr < b.value
    if gate and not good:
        return None, None, None
    try:
        x_bar, i_max = 1 / a.value, 0.5 * np.log2(b.value / a.value)
    except Exception:
        x_bar, i_max = None, None
    return x_bar, i_max, result


def plot_precomputed_scaling(ax, df, param_df, color_map, marker_map, hue_order_metrics, hue_order_methods):
    """Plot precomputed scaling-collapse data (panel b)."""
    n_curves = 0
    for sig in hue_order_metrics:
        if sig == "Caltech101-binary":
            continue
        for q in df["quality"].unique():
            if q < 0.1:
                continue
            for alg in hue_order_methods:
                data = df[(df["signal"] == sig) & (df["quality"] == q) & (df["algorithm"] == alg)]
                if len(data) < 10:
                    continue
                avg_data = data.groupby("size").mi_value.mean()
                p = param_df[(param_df["metric"] == sig) & (param_df["method"] == alg) & (param_df["quality"] == q)]
                if p.empty:
                    continue
                s, N0, I_inf = p["s"].values[0], p["N0"].values[0], p["I_inf"].values[0]
                N_hat = avg_data.index / N0
                plot_points(ax, N_hat, (I_inf - avg_data.values) ** (-1 / s),
                            color_map[alg], None, marker_map[sig])
                n_curves += 1
    print(f"Plotted {n_curves} scaling curves.")


def plot_precomputed_noise(ax, df, param_df, palette, hue_order_metrics, hue_order_methods):
    """Plot precomputed noise-collapse data (panel e), coloured by metric."""
    n_curves = 0
    for idx, sig in enumerate(hue_order_metrics):
        if sig == "Caltech101-binary":
            continue
        for size in df["size"].unique():
            for alg in hue_order_methods:
                data = df[(df["signal"] == sig) & (df["size"] == size) & (df["algorithm"] == alg)]
                if len(data) < 10:
                    continue
                avg_data = data.groupby("umis_per_cell").mi_value.mean()
                p = param_df[(param_df["metric"] == sig) & (param_df["method"] == alg) & (param_df["size"] == size)]
                if p.empty:
                    continue
                xbar, imax = p["fitted_u_bar"].values[0], p["fitted_I_max"].values[0]
                z = transform_to_z(avg_data.index, xbar, imax)
                plot_points(ax, z, avg_data.values - imax, palette[idx], sig)
                n_curves += 1
    print(f"Plotted {n_curves} noise curves.")


def fit_and_plot_caltech(ax, df, x_col, color):
    for class_label in df["Class label"].unique()[:-1]:
        data = df[df["Class label"] == class_label]
        x_data, y_data = 1 / data[x_col], data["MI"]
        x_bar, i_max, _ = fit_info_model(x_data, y_data)
        if x_bar and i_max:
            z = transform_to_z(x_data, x_bar, i_max)
            plot_points(ax, z, y_data - i_max, color, "Caltech101-binary")


def fit_and_plot_sequences(ax, seq_df, color):
    for model_size in sorted(seq_df["model_size"].unique()):
        data = seq_df[seq_df["model_size"] == model_size]
        x_data = data["true/error"]
        y_data = data["mutual_information"]
        x_bar, i_max, _ = fit_info_model(x_data, y_data)
        if x_bar and i_max:
            z = transform_to_z(x_data, x_bar, i_max)
            plot_points(ax, z, y_data - i_max, color, "Spike seqs.")
            print(f"Sequences {model_size}: x_bar={x_bar:.2f}, i_max={i_max:.2f}")


def fit_and_plot_tissue(ax, combined_df, downsampling_type, tissue_color):
    data_subset = combined_df[combined_df["downsampling_type"] == downsampling_type].copy()
    x_transform = (lambda x: x ** 2) if downsampling_type == "pixel" else (lambda x: x)
    data_subset["inv_factor"] = 1 / data_subset["downsampling_level"]
    ova_columns = ["mi_score"] + [c for c in data_subset.columns if "ova_mi_continuous" in c]
    for col in ova_columns[::-1]:
        mask = ~data_subset[col].isna() & ~data_subset["inv_factor"].isna()
        x_data = x_transform(data_subset[mask]["inv_factor"].values)
        y_data = data_subset[mask][col].values
        if len(x_data) < 3:
            continue
        x_bar, i_max, _ = fit_info_model(x_data, y_data)
        if x_bar and i_max:
            z = transform_to_z(x_data, x_bar, i_max)
            plot_points(ax, z, y_data - i_max, tissue_color, "TissueMNIST")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data():
    print("Loading data ...")
    collect_raw = pd.read_csv(COLLECT_CSV)

    # --- fit STATE scaling params and persist them into the param CSVs ---
    print("Fitting STATE scaling parameters ...")
    state_cell = fit_state_cell_scaling(collect_raw)
    state_noise = fit_state_noise_scaling(collect_raw)
    cell_param_raw = merge_and_persist(CELL_SCALING_CSV, state_cell, ["dataset", "method", "metric", "quality"])
    noise_param_raw = merge_and_persist(NOISE_SCALING_CSV, state_noise, ["dataset", "method", "metric", "size"])

    # --- single-cell MI results, renamed like the notebook ---
    df = collect_raw.replace(RENAME_DICT)
    sc_param_df = cell_param_raw.replace(RENAME_DICT)
    sc_param_df_noise = noise_param_raw.replace(RENAME_DICT)

    # --- Caltech101 (remote); skip gracefully if offline ---
    gaussian_df, res_df = None, None
    try:
        gaussian_df = pd.read_csv(f"{CALTECH_URL}Caltech101_Gaussian.csv")
        gaussian_df["Scale"] = gaussian_df["Scale"] ** 2
        res_df = pd.read_csv(f"{CALTECH_URL}Caltech101_resolution.csv")
    except Exception as e:
        print(f"WARNING: could not fetch Caltech101 data ({e}); panel e will omit Caltech points.")

    # --- ESM2 spike sequences ---
    seq_df = pd.read_csv(SEQ_CSV)

    # --- TissueMNIST ---
    dfs_tissue = []
    for file in glob.glob(TISSUE_GLOB):
        match = re.search(r"result_(.+)\.csv", file)
        token = match.group(1)
        if token == "clean":
            downsampling_type, downsampling_level = "clean", 0.0
        elif "pix" in token:
            downsampling_type, downsampling_level = "pixel", float(token.split("_")[1][:-1])
        elif "gauss" in token:
            downsampling_type, downsampling_level = "gaussian", float(token.split("_")[1][:-1])
        else:
            downsampling_type, downsampling_level = "unknown", 0.0
        df_temp = pd.read_csv(file)
        df_temp["downsampling_level"] = downsampling_level
        df_temp["downsampling_type"] = downsampling_type
        dfs_tissue.append(df_temp)
    combined_df = pd.concat(dfs_tissue, ignore_index=True)

    return dict(df=df, sc_param_df=sc_param_df, sc_param_df_noise=sc_param_df_noise,
                noise_param_raw=noise_param_raw, gaussian_df=gaussian_df, res_df=res_df,
                seq_df=seq_df, combined_df=combined_df)


# ----------------------------------------------------------------------------
# Style + colours
# ----------------------------------------------------------------------------
def setup_style():
    sns.set_style("whitegrid")
    rcParams.update({
        "figure.dpi": 150,
        "grid.linewidth": 0.5,
        "grid.linestyle": "dashed",
        "legend.fancybox": False,
        "mathtext.fontset": "stix",
    })
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"


# c6[0] (lavender) is STATE's colour.
c6 = ["#c4bae2", "#40ada6", "#3c4ebf", "#e3a600", "#d6543a", "#edb1a3"]

# Single-cell model -> colour (after RENAME_DICT: RandomProjection -> "Rand. Proj.").
METHOD_COLOR = {
    "Rand. Proj.": c6[1],
    "PCA": c6[2],
    "SCVI": c6[3],
    "Geneformer": c6[4],
    "State": c6[0],   # lavender
}
METHOD_LABEL = {"State": "STATE"}  # display label for the legend

# Model orders, now including STATE.
HUE_ORDER = ["Rand. Proj.", "PCA", "SCVI", "Geneformer", "State"]
PRETTY_PALETTE = [METHOD_COLOR[m] for m in HUE_ORDER]
SCALING_METHODS = ["PCA", "SCVI", "Geneformer", "State"]   # panel b
NOISE_METHODS = ["PCA", "SCVI", "Geneformer", "Rand. Proj.", "State"]  # panel e

# Metric colours / order (unchanged from the notebook).
PRETTY_PALETTE_METRICS = "grey, #C8102EFF, #FF6720FF, #FFC72CFF, #009739FF, #1D4289FF, #702F8AFF".split(", ")
HUE_ORDER_METRICS = ["Protein MI", "Clonal MI", "Temporal MI", "Spatial MI", "Caltech101-binary"]

SEQ_COLORS = "#1D457FFF, #61599DFF, #C36377FF, #EB7F54FF, #F2AF4AFF".split(", ")
SEQ_COLORS = SEQ_COLORS[:1] + SEQ_COLORS[3:4] + SEQ_COLORS[4:]
SEQ_COLOR_COLLAPSE = PRETTY_PALETTE_METRICS[6]
TISSUE_COLOR = PRETTY_PALETTE_METRICS[5]

PALETTE_TISSUE = ("#CBC106FF, #27993CFF, #1C6838FF, #8EBCB5FF, #389CA7FF, "
                  "#4D83ABFF, #CB7B26FF, #BF565DFF, #9E163CFF").split(", ")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(data):
    df = data["df"]
    sc_param_df = data["sc_param_df"]
    sc_param_df_noise = data["sc_param_df_noise"]
    noise_param_raw = data["noise_param_raw"]
    gaussian_df, res_df = data["gaussian_df"], data["res_df"]
    seq_df, combined_df = data["seq_df"], data["combined_df"]

    fig = plt.figure(figsize=(10, 15), dpi=150)
    gs_main = fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 2, 3], hspace=0.75)

    # ========================================================================
    # ROW 1: Cell Number Scaling
    # ========================================================================
    gs_row1 = gs_main[0].subgridspec(1, 2, wspace=0.4)

    # Panel a: scaling example (Temporal MI, Geneformer)
    ax_a = fig.add_subplot(gs_row1[0])
    metric, method = "Temporal MI", "Geneformer"
    subset = sc_param_df[(sc_param_df["quality"] == 1) & (sc_param_df["method"] == method) & (sc_param_df["metric"] == metric)]
    s, N0, I_inf = subset["s"].values[0], subset["N0"].values[0], subset["I_inf"].values[0]

    data_a = df[(df["quality"] == 1) & (df["algorithm"] == method) & (df["signal"] == metric)]
    sns.scatterplot(data=data_a, x="size", y="mi_value", color=METHOD_COLOR["Geneformer"], ax=ax_a, legend=False, zorder=3)

    xs = np.logspace(1, 7, 100)
    ys = np.maximum(I_inf - (xs / N0) ** (-s), 0)
    ax_a.plot(xs, ys, linestyle=":", color=METHOD_COLOR["Geneformer"], zorder=20)

    ax_a.axhline(I_inf, linestyle="dashed", color="darkcyan", zorder=10, lw=1.5)
    ax_a.text(data_a["size"].max() / 2, I_inf - 0.5, r"capacity $I_\infty$", ha="right", va="bottom", color="darkcyan", fontsize=9)
    ax_a.axvline(N0, linestyle="dashed", color="darkgreen", zorder=10, lw=1.5)
    ax_a.text(N0 * 2, 0.5, r"1-bit saturation point $N_0$", ha="left", va="bottom", color="darkgreen", fontsize=9)

    y_tangent = I_inf - 1
    x_tan_1, x_tan_2 = N0 / 8, N0 * 16
    y_tan_1 = s * np.log(x_tan_1) - s * np.log(N0) + y_tangent
    y_tan_2 = s * np.log(x_tan_2) - s * np.log(N0) + y_tangent
    ax_a.plot([x_tan_1, x_tan_2], [y_tan_1, y_tan_2], linestyle="dashed", color="purple", zorder=10, lw=1.5)
    ax_a.text(x_tan_2 / 4, y_tan_2 - 1, r" scaling exponent $s$", ha="left", va="center", color="purple", fontsize=9)

    ax_a.set_xscale("log")
    ax_a.set_xlabel("Cell number")
    ax_a.set_ylabel("Temporal MI (bits)")
    ax_a.set_xlim(data_a["size"].min() * 0.8, data_a["size"].max() * 1.2)

    # Panel b: scaling collapse
    ax_b = fig.add_subplot(gs_row1[1])
    markers = ["o", "s", "^", "D", "v"]
    marker_map = {m: markers[i] for i, m in enumerate(HUE_ORDER_METRICS) if m != "Caltech101-binary"}

    plot_precomputed_scaling(ax_b, df, sc_param_df, METHOD_COLOR, marker_map, HUE_ORDER_METRICS, SCALING_METHODS)

    xs = np.logspace(-4, 6, 100)
    ax_b.plot(xs, xs, color="black", linestyle="--", alpha=0.4, lw=1.5, label=r"$y = x$", zorder=10)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$N/N_0$", fontsize=16)
    ax_b.set_ylabel(r"$(I_{\infty} - I)^{-1/s}$", fontsize=16)

    method_handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker="o", linestyle="None",
                             label=METHOD_LABEL.get(m, m), ms=4) for m in SCALING_METHODS]
    leg1 = ax_b.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(1.02, 1.15), title="model")
    metric_handles = [Line2D([0], [0], color="black", marker=marker_map[m], linestyle="None", label=m, ms=4)
                      for m in HUE_ORDER_METRICS if m != "Caltech101-binary"]
    ax_b.legend(handles=metric_handles, loc="upper left", bbox_to_anchor=(1.02, 0.42), title="metric")
    ax_b.add_artist(leg1)

    # ========================================================================
    # ROW 2: Noise Examples
    # ========================================================================
    gs_row2 = gs_main[1].subgridspec(1, 4, wspace=0.5)
    axs_c = [fig.add_subplot(gs_row2[i]) for i in range(4)]

    for ax, sig in zip(axs_c, HUE_ORDER_METRICS):
        if sig == "Caltech101-binary":
            continue
        sub = df[df["signal"] == sig]
        max_sub = sub[sub["size"] == sub["size"].max()]

        sns.scatterplot(data=max_sub, x="umis_per_cell", y="mi_value", hue="algorithm",
                        palette=PRETTY_PALETTE, hue_order=HUE_ORDER, ax=ax, legend=ax == axs_c[-1])

        for i, alg in enumerate(HUE_ORDER):
            dat = max_sub[max_sub["algorithm"] == alg]
            if len(dat) < 9:
                continue
            # gate=False: always draw the example fit, regardless of its uncertainty
            _, _, result = fit_info_model(dat["umis_per_cell"], dat["mi_value"], gate=False)
            if result:
                x_fit = np.linspace(dat["umis_per_cell"].min() / 5, dat["umis_per_cell"].max() * 5, 10000)
                y_fit = result.eval(x=x_fit)
                y_err = result.eval_uncertainty(x=x_fit, sigma=2)
                ax.fill_between(x_fit, y_fit + y_err, y_fit - y_err, color=PRETTY_PALETTE[i], alpha=0.2)
                ax.plot(x_fit, y_fit, color=PRETTY_PALETTE[i], linestyle="--")

        ax.set_xscale("log")
        ax.set_ylabel(sig, fontsize=12)
        ax.set_xlabel("UMI per cell", fontsize=12)

    if axs_c[-1].get_legend():
        # relabel STATE -> "STATE" in the model legend
        leg = axs_c[-1].get_legend()
        for txt in leg.get_texts():
            if txt.get_text() == "State":
                txt.set_text("STATE")
        sns.move_legend(axs_c[-1], "upper left", bbox_to_anchor=(1, 1), title="model")
        for txt in axs_c[-1].get_legend().get_texts():
            if txt.get_text() == "State":
                txt.set_text("STATE")

    # ========================================================================
    # ROW 3: Noise Collapse
    # ========================================================================
    gs_row3 = gs_main[2].subgridspec(1, 2, width_ratios=[4, 4], wspace=0.5)

    # Panel d: noise annotation (Temporal MI, Geneformer)
    ax_d = fig.add_subplot(gs_row3[0])
    metric, method = "Temporal MI", "Geneformer"
    max_cell_number = df[df["signal"] == metric]["size"].max()
    subset = sc_param_df_noise[(sc_param_df_noise["size"] == max_cell_number) &
                               (sc_param_df_noise["method"] == method) & (sc_param_df_noise["metric"] == metric)]
    u_bar, I_max = subset["fitted_u_bar"].values[0], subset["fitted_I_max"].values[0]

    data_d = df[(df["size"] == max_cell_number) & (df["algorithm"] == method) & (df["signal"] == metric)]
    sns.scatterplot(data=data_d, x="umis_per_cell", y="mi_value", color=METHOD_COLOR["Geneformer"], ax=ax_d, legend=False, zorder=3)

    xs = np.logspace(np.log10(data_d["umis_per_cell"].min() / 2), np.log10(data_d["umis_per_cell"].max() * 2), 100)
    ys = np.maximum(I_max - 0.5 * np.log2((1 + xs / u_bar) / (xs / u_bar + 2 ** (-2 * I_max))), 0)
    ax_d.plot(xs, ys, linestyle=":", color=METHOD_COLOR["Geneformer"], zorder=20)

    ax_d.axhline(I_max, linestyle="dashed", color="darkcyan", zorder=10, lw=1.5)
    ax_d.text(data_d["umis_per_cell"].max() / 2, I_max + 0.1, r"capacity $I_\max$", ha="center", va="bottom", color="darkcyan", fontsize=9)
    ax_d.axvline(u_bar, linestyle="dashed", color="darkgreen", zorder=10, lw=1.5)
    ax_d.text(u_bar * 1.25, 0.5, r"sensitivity $\bar{\eta}$", ha="left", va="bottom", color="darkgreen", fontsize=9)

    ax_d.set_xscale("log")
    ax_d.set_ylim(0, 2.5)
    ax_d.set_xlabel("UMI per cell")
    ax_d.set_ylabel("Temporal MI (bits)")

    # Panel e: noise collapse (coloured by metric; STATE adds points here too)
    ax_e = fig.add_subplot(gs_row3[1])
    plot_precomputed_noise(ax_e, df, sc_param_df_noise, PRETTY_PALETTE_METRICS, HUE_ORDER_METRICS, NOISE_METHODS)
    if gaussian_df is not None:
        fit_and_plot_caltech(ax_e, gaussian_df, "Scale", PRETTY_PALETTE_METRICS[4])
        fit_and_plot_caltech(ax_e, res_df, "Factor", PRETTY_PALETTE_METRICS[4])
    fit_and_plot_tissue(ax_e, combined_df, "pixel", TISSUE_COLOR)
    fit_and_plot_tissue(ax_e, combined_df, "gaussian", TISSUE_COLOR)
    fit_and_plot_sequences(ax_e, seq_df, SEQ_COLOR_COLLAPSE)

    xs = np.logspace(-6, 5.5, 100)
    ax_e.plot(xs, -np.log2((1 + xs) / xs) / 2, color="black", linestyle="--", alpha=0.4, lw=1.5,
              label=r"$y = -\frac{1}{2}\log(1 + z^{-1})$", zorder=10)
    ax_e.set_xscale("log")
    ax_e.set_xlim(xs.min(), xs.max())
    ax_e.set_ylabel(r"$\mathcal{I} - \mathcal{I}_{\max}$", fontsize=16)
    ax_e.set_xlabel(r"$z = \frac{\eta/\bar{\eta} + 2^{-2\mathcal{I}_{\max}}}{1 - 2^{-2\mathcal{I}_{\max}}}$", fontsize=16)

    handles, labels = ax_e.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_e.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0), title="metric")

    # ========================================================================
    # ROW 4: Parameter Bars
    # ========================================================================
    gs_row4 = gs_main[3].subgridspec(1, 2, wspace=0.3)
    axs_f = [fig.add_subplot(gs_row4[i]) for i in range(2)]

    # Error cutoffs for the parameter bars.  The capacity (I_max) bars are gated
    # on the absolute capacity error; the sensitivity (u_bar) bars are gated on
    # the *relative* sensitivity error -- a poorly-constrained u_bar (e.g. Clonal
    # MI STATE, ~55% relative error) is suppressed even when its capacity is well
    # constrained, so the two bars come from the same fit but are filtered on the
    # parameter each panel actually shows.
    I_MAX_ERR_CUTOFF = 0.5          # capacity error, in bits
    U_BAR_REL_ERR_CUTOFF = 0.5      # sensitivity error, as a fraction of u_bar

    noise_df = noise_param_raw.copy()
    noise_df = noise_df[noise_df["I_max_error"] < I_MAX_ERR_CUTOFF]
    noise_df = noise_df[noise_df["metric"] != "celltype.l3"]
    noise_df = noise_df[noise_df["size"].isin(noise_df.groupby("dataset")["size"].max().values)]

    # Sensitivity panel: suppress (NaN out) bars whose u_bar is poorly constrained,
    # keeping the metric x method grid aligned with the capacity panel.
    sens_df = noise_df.copy()
    rel_ubar_err = sens_df["u_bar_error"] / sens_df["fitted_u_bar"]
    sens_df.loc[rel_ubar_err >= U_BAR_REL_ERR_CUTOFF, ["fitted_u_bar", "u_bar_error"]] = np.nan

    noise_df.replace(RENAME_DICT, inplace=True)
    sens_df.replace(RENAME_DICT, inplace=True)

    bar_methods = [m for m in HUE_ORDER if m in noise_df["method"].unique()]
    bar_colors = [METHOD_COLOR[m] for m in bar_methods]

    means = sens_df.pivot(index="metric", columns="method", values="fitted_u_bar")
    errors = sens_df.pivot(index="metric", columns="method", values="u_bar_error")
    means[bar_methods].plot(kind="bar", yerr=errors[bar_methods], ax=axs_f[0], color=bar_colors,
                            capsize=0, ecolor="grey", rot=0, legend=False)
    axs_f[0].set_xlabel("")
    axs_f[0].set_ylabel(r"sensitivity ($\bar{\eta}$)", fontsize=12)
    axs_f[0].tick_params(axis="x", rotation=60)

    means = noise_df.pivot(index="metric", columns="method", values="fitted_I_max")
    errors = noise_df.pivot(index="metric", columns="method", values="I_max_error")
    means[bar_methods].plot(kind="bar", yerr=errors[bar_methods], ax=axs_f[1], color=bar_colors,
                            capsize=0, ecolor="grey", rot=0)
    axs_f[1].set_xlabel("")
    axs_f[1].set_ylabel(r"capacity ($\mathcal{I}_{\max}$)", fontsize=12)
    leg_f = axs_f[1].legend(title="model", bbox_to_anchor=(1.0, 1), loc="upper left")
    for txt in leg_f.get_texts():
        if txt.get_text() == "State":
            txt.set_text("STATE")
    axs_f[1].tick_params(axis="x", rotation=60)

    # ========================================================================
    # ROW 5: TissueMNIST and Sequences
    # ========================================================================
    gs_row5 = gs_main[4].subgridspec(1, 3, wspace=0.3)
    axs_g = [fig.add_subplot(gs_row5[i]) for i in range(3)]

    label_map = {
        "ova_mi_continuous_Class_0": "Collecting Duct, Connecting Tubule",
        "ova_mi_continuous_Class_1": "Distal Convoluted Tubule",
        "ova_mi_continuous_Class_2": "Glomerular endothelial cells",
        "ova_mi_continuous_Class_3": "Interstitial endothelial cells",
        "ova_mi_continuous_Class_4": "Leukocytes",
        "ova_mi_continuous_Class_5": "Podocytes",
        "ova_mi_continuous_Class_6": "Proximal Tubule Segments",
        "ova_mi_continuous_Class_7": "Thick Ascending Limb",
        "mi_score": "8-class MI",
    }

    # Panel g1: sequences
    for idx, model_size in enumerate(sorted(seq_df["model_size"].unique())):
        data_s = seq_df[seq_df["model_size"] == model_size]
        x_data = data_s["true/error"]
        y_data = data_s["mutual_information"]
        color = SEQ_COLORS[idx % len(SEQ_COLORS)]
        _, _, result = fit_info_model(x_data, y_data)
        if result:
            x_fit = np.logspace(np.log10(x_data.min() / 5), np.log10(5 * x_data.max()), 100000)
            y_fit = result.eval(x=x_fit)
            y_err = result.eval_uncertainty(x=x_fit, sigma=2)
            axs_g[0].plot(x_fit, y_fit, linestyle="--", color=color, alpha=0.5)
            axs_g[0].fill_between(x_fit, y_fit + y_err, y_fit - y_err, color=color, alpha=0.2)
        axs_g[0].scatter(x_data, y_data, color=color, s=30, label=f"ESM2-{model_size}")
    axs_g[0].set_xlabel(r"$\frac{\text{true a.a.}}{\text{corrupted a.a.}}$", fontsize=14)
    axs_g[0].set_ylabel(r"$I(L; z)$", fontsize=14)
    axs_g[0].set_xscale("log")
    axs_g[0].legend(loc="lower right", fontsize=8)

    # Panel g2: pixel downsampling
    pix = combined_df[combined_df["downsampling_type"] == "pixel"].copy()
    pix["inv_factor"] = 1 / pix["downsampling_level"]
    ova_columns = ["mi_score"] + [c for c in pix.columns if "ova_mi_continuous" in c]
    for i, col in enumerate(ova_columns[::-1]):
        mask = ~pix[col].isna() & ~pix["inv_factor"].isna()
        x_data, y_data = pix[mask]["inv_factor"].values ** 2, pix[mask][col].values
        _, _, result = fit_info_model(x_data, y_data)
        if result:
            x_fit = np.logspace(np.log10(x_data.min() / 5), np.log10(5 * x_data.max()), 100000)
            y_fit = result.eval(x=x_fit)
            y_err = result.eval_uncertainty(x=x_fit, sigma=2)
            axs_g[1].plot(x_fit, y_fit, linestyle="--", color=PALETTE_TISSUE[i], alpha=0.5)
            axs_g[1].fill_between(x_fit, y_fit + y_err, y_fit - y_err, color=PALETTE_TISSUE[i], alpha=0.2)
        axs_g[1].scatter(x_data, y_data, color=PALETTE_TISSUE[i], label=label_map.get(col, col), s=30)
    axs_g[1].set_xlabel(r"$(\text{downsampling factor})^{-1}$", fontsize=14)
    axs_g[1].set_xscale("log")

    # Panel g3: gaussian noise
    gauss = combined_df[combined_df["downsampling_type"] == "gaussian"].copy()
    for i, col in enumerate(ova_columns[::-1]):
        mask = ~gauss[col].isna()
        x_data, y_data = 1 / gauss[mask]["downsampling_level"].values, gauss[mask][col].values
        _, _, result = fit_info_model(x_data, y_data)
        if result:
            x_fit = np.logspace(np.log10(x_data.min() / 5), np.log10(5 * x_data.max()), 100000)
            y_fit = result.eval(x=x_fit)
            y_err = result.eval_uncertainty(x=x_fit, sigma=2)
            axs_g[2].plot(x_fit, y_fit, linestyle="--", color=PALETTE_TISSUE[i], alpha=0.5)
            axs_g[2].fill_between(x_fit, y_fit + y_err, y_fit - y_err, color=PALETTE_TISSUE[i], alpha=0.2)
        axs_g[2].scatter(x_data, y_data, color=PALETTE_TISSUE[i], label=label_map.get(col, col), s=30)
    axs_g[2].set_xlabel(r"$(\text{variance})^{-1}$", fontsize=14)
    axs_g[2].set_xscale("log")
    axs_g[2].legend(loc="upper left", bbox_to_anchor=(1, 1.05), fontsize=8)

    # ========================================================================
    # Panel labels
    # ========================================================================
    for ax, label in zip([ax_a, ax_b, axs_c[0], ax_d, ax_e, axs_f[0], axs_g[0]],
                         ["a)", "b)", "c)", "d)", "e)", "f)", "g)"]):
        x_pos = -0.3 if ax != axs_c[0] else -0.62
        ax.text(x_pos, 1.05, label, transform=ax.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {os.path.relpath(OUT_PDF, REPO_ROOT)} and {os.path.relpath(OUT_PNG, REPO_ROOT)}")


def main():
    setup_style()
    data = load_data()
    make_figure(data)


if __name__ == "__main__":
    main()
