import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "final_results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

sns.set_style("whitegrid")
rcParams["figure.dpi"] = 150
rcParams["grid.linewidth"] = 0.5
rcParams["grid.linestyle"] = "dashed"
rcParams["legend.fancybox"] = False
rcParams["mathtext.fontset"] = "stix"

RENAME = {
    "celltype.l3": "Cell type MI",
    "protein_counts": "Protein MI",
    "clone": "Clonal MI",
    "author_day": "Temporal MI",
    "ng_idx": "Spatial MI",
    "RandomProjection": "Rand. Proj.",
    "State": "STATE",
}

metric_order = ["Cell type MI", "Protein MI", "Clonal MI", "Temporal MI", "Spatial MI"]
metric_palette = ["#C85158", "#62A48F", "#475A7A", "#503A38", "#E9AD97"]

c6 = ["#c4bae2", "#40ada6", "#3c4ebf", "#e3a600", "#d6543a", "#edb1a3"]
method_color = {"Rand. Proj.": c6[1], "PCA": c6[2], "SCVI": c6[3],
                "Geneformer": c6[4], "STATE": c6[0]}


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv(os.path.join(DATA, "collect_mi_results.csv")).replace(RENAME)
sc_param_df = pd.read_csv(os.path.join(DATA, "cell_scaling.csv")).replace(RENAME)

sc_param_df["min I"] = [df[(df["signal"] == m) & (df["quality"] == 1) & (df["algorithm"] == meth)]["mi_value"].min()
                        for m, meth in zip(sc_param_df["metric"], sc_param_df["method"])]


# N90: cells to reach 90% of I_inf.  Bits are logarithmic, so the captured
# fraction is 2^-(I_inf - I); a 1-bit deficit (= N0) is 50%.  For 90% the
# deficit is a fixed -log2(0.9) bits, independent of I_inf:
#   (N90/N0)^(-s) = -log2(0.9)  =>  N90 = N0 * (-log2(0.9))^(-1/s)
# ln N90 = ln N0 - (1/s) ln(DELTA)  =>  only N0 and s carry error.
DELTA90 = -np.log2(0.9)  # ~0.152 bits below saturation
with np.errstate(divide="ignore", invalid="ignore"):
    sc_param_df["N90"] = sc_param_df["N0"] * DELTA90 ** (-1.0 / sc_param_df["s"])
    _var_ln = ((sc_param_df["N0_error"] / sc_param_df["N0"]) ** 2
               + ((np.log(DELTA90) / sc_param_df["s"] ** 2) * sc_param_df["s_error"]) ** 2)
    sc_param_df["N90_error"] = sc_param_df["N90"] * np.sqrt(_var_ln)


# ranked scaling exponent s and capacity I_inf (quality=1, deep methods)
subset = sc_param_df[(sc_param_df["quality"] == 1) & (sc_param_df["method"] != "Rand. Proj.")]
style_order = ["Geneformer", "PCA", "SCVI", "STATE"]

fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for ax, col in zip(axs, ["s", "I_inf"]):
    ranked = subset.sort_values(col, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    sns.scatterplot(data=ranked, x="rank", y=col, hue="metric", palette=metric_palette,
                    style="method", s=60, edgecolor="w", ax=ax,
                    hue_order=metric_order, style_order=style_order,
                    legend=(col == "I_inf"))
axs[0].set_ylabel(r"$s$ (scaling exponent)")
axs[1].set_ylabel(r"$I_\infty$ (capacity)")
axs[1].set_ylim(0)
axs[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
save(fig, "cell_scaling_ranked_params")


# scaling exponent s vs capacity I_inf
fig, ax = plt.subplots(figsize=(4, 3))
sns.scatterplot(data=subset, x="s", y="I_inf", hue="metric", palette=metric_palette,
                style="method", s=60, edgecolor="w", ax=ax,
                hue_order=metric_order, style_order=style_order)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xlabel(r"$s$ (scaling exponent)")
ax.set_ylabel(r"$I_\infty$ (saturation point)")
ax.set_xlim(0)
ax.set_ylim(0)
fig.tight_layout()
save(fig, "cell_scaling_s_vs_Iinf")


# temporal MI scatter and fit per model
metric = "Temporal MI"
fig, ax = plt.subplots(figsize=(4, 3))
for method in ["Rand. Proj.", "PCA", "SCVI", "Geneformer", "STATE"]:
    p = sc_param_df[(sc_param_df["quality"] == 1) & (sc_param_df["method"] == method)
                    & (sc_param_df["metric"] == metric)]
    if p.empty:
        continue
    data = df[(df["quality"] == 1) & (df["algorithm"] == method) & (df["signal"] == metric)]
    color = method_color[method]
    ax.scatter(data["size"], data["mi_value"], color=color, s=15, label=method)
    s, N0, I_inf = p["s"].values[0], p["N0"].values[0], p["I_inf"].values[0]
    xs = np.logspace(1, 7, 100)
    ax.plot(xs, np.maximum(I_inf - (xs / N0) ** -s, 0), linestyle="--", color=color)
ax.set_xscale("log")
ax.set_xlabel("Cell number")
ax.set_ylabel("Temporal information (bits)")
ax.legend(title="model", bbox_to_anchor=(1, 1), loc="upper left")
fig.tight_layout()
save(fig, "cell_scaling_temporal_fit")


# capacity / saturation / sensitivity strips per metric (quality=1, deep methods)
color_dict = {m: method_color[m] for m in ["Geneformer", "PCA", "SCVI", "STATE"]}
S_ERR_MAX = 5.0  # sensitivity error bars wider than this are omitted (they blow up the y-axis)
N90_ERR_MAX_REL = 3.0  # N90 error bars with rel. size above this are omitted (log axis)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
x_count = 0
centers, labels = [], []
for metric in sc_param_df["metric"].unique():
    if metric == "Cell type MI":
        continue
    sub = sc_param_df[(sc_param_df["quality"] == 1) & (sc_param_df["metric"] == metric)].copy()
    sub["I_inf - min I"] = sub["I_inf"] - sub["min I"]
    sub = sub[(sub["I_inf - min I"] > 0.2) & (sub["mean_residual"] < 0.75)]
    sub = sub[~sub["method"].isin(["Rand. Proj."])]
    if sub.empty:
        continue
    n = len(sub)
    positions = x_count + np.arange(n)
    centers.append(x_count + (n - 1) / 2.0)
    labels.append(metric[:-3])
    for ax, col, err in [(axs[0], "I_inf", "I_inf_error"),
                         (axs[1], "N0", "N0_error"),
                         (axs[2], "s", "s_error"),
                         (axs[3], "N90", "N90_error")]:
        s = sub.sort_values(col, ascending=False)
        colors = np.array([color_dict.get(m, "black") for m in s["method"]])
        ax.scatter(positions, s[col], color=colors, zorder=3, s=40)
        keep = np.ones(n, dtype=bool)
        if col == "s":
            keep = s[err].values <= S_ERR_MAX
            for xp, yp, c in zip(positions[~keep], s[col].values[~keep], colors[~keep]):
                ax.annotate("x", (xp, yp), textcoords="offset points", xytext=(0, -9),
                            ha="center", va="center", fontsize=7, color=c)
        elif col == "N90":
            keep = (s[err].values / s[col].values) <= N90_ERR_MAX_REL
        ax.errorbar(positions[keep], s[col].values[keep], yerr=s[err].values[keep],
                    fmt="none", ecolor=colors[keep], zorder=2, alpha=0.7)
    for ax in axs:
        ax.axvline(x=x_count + n, color="black", alpha=0.5, lw=1)
    x_count += n + 1

for ax in axs:
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=9)
    ax.set_xlim(-1, x_count - 1)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
axs[0].set_ylabel(r"capacity ($I_{\infty}$)", fontsize=12)
axs[1].set_ylabel(r"saturation ($N_{0}$)", fontsize=12)
axs[1].set_yscale("log")
axs[2].set_ylabel(r"sensitivity ($s$)", fontsize=12)
axs[3].set_ylabel(r"90% saturation ($N_{90}$)", fontsize=12)
axs[3].set_yscale("log")
axs[3].legend(handles=[plt.Line2D([0], [0], marker="o", color="w", label=m,
                                  markersize=8, markerfacecolor=color_dict[m])
                       for m in ["Geneformer", "PCA", "SCVI", "STATE"]],
              title="model", fontsize=10, bbox_to_anchor=(1.02, 1.05), loc="upper left")
fig.tight_layout()
save(fig, "cell_scaling_param_strips")


# R^2 of the cell number scaling fits (MI space)
def compute_r2(df, param_df):
    rows = []
    for sig in df["signal"].unique():
        if sig == "Cell type MI":
            continue
        for q in df["quality"].unique():
            for alg in ["PCA", "Geneformer", "SCVI", "STATE"]:
                data = df[(df["signal"] == sig) & (df["quality"] == q) & (df["algorithm"] == alg)]
                if len(data) < 10:
                    continue
                avg = data.groupby("size").mi_value.mean()
                x, y = avg.index.values, avg.values
                p = param_df[(param_df["metric"] == sig) & (param_df["method"] == alg) & (param_df["quality"] == q)]
                if p.empty:
                    continue
                s, N0, I_inf = p["s"].values[0], p["N0"].values[0], p["I_inf"].values[0]
                y_pred = I_inf - (x / N0) ** (-s)
                rows.append({"signal": sig, "quality": q, "method": alg,
                             "r2": r2_score(y, y_pred), "n_points": len(y)})
    return pd.DataFrame(rows)


r2_df = compute_r2(df, sc_param_df)
print(r2_df[r2_df["quality"] == 1].to_string())

sel = r2_df[(r2_df["method"].isin(["Geneformer", "SCVI", "STATE"])) & (r2_df["quality"] >= 0.1)]
print(f"mean R^2: {sel['r2'].mean():.4f}")
print(f"sem: {sel['r2'].sem():.4f}")
print(f"n: {len(sel)}")
negatives = sel[sel["r2"] < 0]
if len(negatives):
    print(f"{len(negatives)} negative R^2 fit(s):")
    print(negatives[["signal", "quality", "method", "r2"]].to_string())
else:
    print("no negative R^2 fits")
