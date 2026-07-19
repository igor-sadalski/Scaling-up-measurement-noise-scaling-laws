import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import utils

datasets = ["PBMC", "larry", "merfish", "shendure"]
mi_label = {"protein_counts": "protein MI", "clone": "clonal MI",
            "ng_idx": "spatial MI", "author_day": "temporal MI"}


def cell_number_scaling(x, N0, s, I_inf):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.maximum(0, I_inf - (x / N0) ** (-s))


df = utils.load_collect()
params = pd.read_csv(utils.path("cell_scaling.csv"))
params["quality"] = params["quality"].round(6)

umis = df["umis_per_cell"].values
norm = mcolors.LogNorm(vmin=umis[umis > 0].min(), vmax=umis.max())
cmap = plt.get_cmap("viridis")

methods = utils.METHOD_ORDER
fig, axes = plt.subplots(len(datasets), len(methods),
                         figsize=(13, 9.5), squeeze=False)

for i, dataset in enumerate(datasets):
    signal = utils.DATASET_SIGNAL[dataset]
    for j, method in enumerate(methods):
        ax = axes[i][j]
        sub = df[(df["dataset"] == dataset) & (df["signal"] == signal)
                 & (df["algorithm"] == method)]
        par = params[(params["dataset"] == dataset) & (params["metric"] == signal)
                     & (params["method"] == method)]
        for q in sorted(sub["quality"].unique()):
            avg = sub[sub["quality"] == q].groupby("size")["mi_value"].mean()
            x, y = avg.index.values, avg.values
            color = cmap(norm(sub[sub["quality"] == q]["umis_per_cell"].iloc[0]))
            ax.scatter(x, y, color=color, s=10)
            p = par[np.isclose(par["quality"], round(q, 6))]
            if not p.empty:
                xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
                yf = cell_number_scaling(xf, p["N0"].iloc[0], p["s"].iloc[0], p["I_inf"].iloc[0])
                ax.plot(xf, yf, linestyle="--", color=color, lw=1)
        ax.set_xscale("log")
        if i == 0:
            ax.set_title(utils.METHOD_LABEL[method])
        if i == len(datasets) - 1:
            ax.set_xlabel("cell number")
        if j == 0:
            ax.set_ylabel(f"{dataset}\n{mi_label[signal]} (bits)")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("UMI per cell")

utils.save(fig, "cell_number_scaling_fits")
