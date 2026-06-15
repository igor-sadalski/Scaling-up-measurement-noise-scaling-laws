import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import utils

datasets = ["PBMC", "larry", "merfish", "shendure"]
mi_label = {"protein_counts": "protein MI", "clone": "clonal MI",
            "ng_idx": "spatial MI", "author_day": "temporal MI"}


def fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    return f"{n/1_000:.0f}k"


df = utils.load_collect()
df = df[(df["algorithm"] == "State") & (df["size"] >= 10000)]

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
r2s = []
for ax, dataset in zip(axes, datasets):
    signal = utils.DATASET_SIGNAL[dataset]
    sub = df[(df["dataset"] == dataset) & (df["signal"] == signal)]
    sizes = sorted(sub["size"].unique())
    cmap = plt.get_cmap("viridis")
    norm = mcolors.LogNorm(vmin=min(sizes), vmax=max(sizes))
    for s in sizes:
        avg = sub[sub["size"] == s].groupby("umis_per_cell")["mi_value"].mean()
        x, y = avg.index.values, avg.values
        color = cmap(norm(s))
        ax.scatter(x, y, color=color, s=10, label=fmt(s))
        if len(x) >= 4:
            result = utils.fit_noise(x, y)
            utils.plot_fit(ax, x, result, color, lw=1)
            r2s.append(result.rsquared)
    ax.set_xscale("log")
    ax.set_xlabel("UMI per cell")
    ax.set_ylabel(f"{mi_label[signal]} (bits)")
    ax.text(0.05, 0.95, dataset, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.legend(title="cell number", fontsize=5, title_fontsize=6, loc="lower right")

print(f"mean R^2 over {len(r2s)} curves: {np.mean(r2s):.3f}")
fig.tight_layout()
utils.save(fig, "02_state_noise_scaling")
