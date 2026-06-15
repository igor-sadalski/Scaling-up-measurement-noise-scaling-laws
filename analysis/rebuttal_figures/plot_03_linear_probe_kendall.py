import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import utils

probe = pd.read_csv(utils.path("linear_probe_scaling.csv"))
probe["q"] = probe["quality"].round(6)

mi = utils.load_collect()
mi = mi[mi.apply(lambda r: r["signal"] == utils.DATASET_SIGNAL.get(r["dataset"]), axis=1)]
mi = mi.groupby(["dataset", "algorithm", "size", "quality"])["mi_value"].mean().reset_index()
mi = mi.rename(columns={"algorithm": "algo"})
mi["q"] = mi["quality"].round(6)

merged = probe.merge(mi[["dataset", "algo", "size", "q", "mi_value"]],
                     on=["dataset", "algo", "size", "q"])

datasets = ["PBMC", "larry", "merfish", "shendure"]
ylabels = {"logreg_accuracy": "linear probe accuracy", "ridge_mean_r2": "linear probe R$^2$"}

fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
taus = []
for ax, dataset in zip(axes, datasets):
    sub = merged[merged["dataset"] == dataset]
    for algo in utils.METHOD_ORDER:
        pts = sub[sub["algo"] == algo]
        ax.scatter(pts["mi_value"], pts["value"], s=8, color=utils.METHOD_COLOR[algo],
                   label=utils.METHOD_LABEL[algo])
    tau = kendalltau(sub["mi_value"], sub["value"]).correlation
    taus.append(tau)
    ax.text(0.05, 0.95, f"{dataset}\ntau={tau:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=7)
    ax.set_xlabel("MI (bits)")
    ax.set_ylabel(ylabels[sub["probe"].iloc[0]])

print(f"mean kendall tau across datasets: {np.mean(taus):.3f}")
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="model")
fig.tight_layout()
utils.save(fig, "03_linear_probe_vs_mi")
