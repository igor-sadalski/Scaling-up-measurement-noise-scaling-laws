import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

datasets = ["PBMC", "larry", "merfish", "shendure"]
mi_label = {"protein_counts": "protein MI", "clone": "clonal MI",
            "ng_idx": "spatial MI", "author_day": "temporal MI"}
metric_color = {"protein_counts": "#C8102EFF", "clone": "#FF6720FF",
                "ng_idx": "#009739FF", "author_day": "#FFC72CFF"}


def cell_number_scaling(x, N0, s, I_inf):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.maximum(0, I_inf - (x / N0) ** (-s))


r2_exclude = ("RandomProjection", "PCA")

df = utils.load_collect()
df = df[np.isclose(df["quality"], 1)]
params = pd.read_csv(utils.path("cell_scaling.csv"))
params = params[np.isclose(params["quality"], 1)]

def r_squared(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


r2s = []
fig, axes = plt.subplots(1, len(utils.METHOD_ORDER), figsize=(13, 2.8))
for ax, method in zip(axes, utils.METHOD_ORDER):
    for dataset in datasets:
        signal = utils.DATASET_SIGNAL[dataset]
        sub = df[(df["dataset"] == dataset) & (df["signal"] == signal)
                 & (df["algorithm"] == method)]
        avg = sub.groupby("size")["mi_value"].mean()
        x, y = avg.index.values, avg.values
        color = metric_color[signal]
        ax.scatter(x, y, color=color, s=10, label=mi_label[signal])
        p = params[(params["dataset"] == dataset) & (params["metric"] == signal)
                   & (params["method"] == method)]
        if not p.empty:
            xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
            yf = cell_number_scaling(xf, p["N0"].iloc[0], p["s"].iloc[0], p["I_inf"].iloc[0])
            ax.plot(xf, yf, linestyle="--", color=color, lw=1)
            if method not in r2_exclude:
                r2 = r_squared(y, cell_number_scaling(x, p["N0"].iloc[0], p["s"].iloc[0], p["I_inf"].iloc[0]))
                r2s.append((f"{method}/{dataset}", r2))
    ax.set_xscale("log")
    ax.set_xlabel("cell number")
    ax.set_ylabel("MI (bits)")
    ax.set_title(utils.METHOD_LABEL[method])

values = [r for _, r in r2s]
print(f"mean R^2 over {len(values)} fits: {np.mean(values):.3f}")
negatives = [(name, r) for name, r in r2s if r < 0]
if negatives:
    print(f"{len(negatives)} negative R^2 fit(s):")
    for name, r in negatives:
        print(f"    {name}: {r:.3f}")
else:
    print("no negative R^2 fits")

axes[-1].legend(title="signal", fontsize=6, title_fontsize=7,
                loc="upper left", bbox_to_anchor=(1.02, 1))
fig.tight_layout()
utils.save(fig, "cell_number_scaling_fits_row")
