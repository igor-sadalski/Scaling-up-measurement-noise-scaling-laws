import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import utils

df = pd.read_csv(utils.path("hyperpam_sweep.csv"))
df = df.dropna(subset=["trial_id"])
df = utils.add_umis(df, "PBMC")

trials = sorted(df["trial_id"].unique())
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
pearsons = []
for i, t in enumerate(trials):
    sub = df[df["trial_id"] == t].sort_values("umis_per_cell")
    x, y = sub["umis_per_cell"].values, sub["mi"].values
    c = sub.iloc[0]
    label = f"lr={c.max_lr:.0e} bs={int(c.batch_size)} do={c.dropout:.1f} wd={c.weight_decay:.0e}"
    color = cmap(i % 10)
    ax.scatter(x, y, color=color, s=10, label=label)
    if y.max() >= 3:
        result = utils.fit_noise(x, y)
        utils.plot_fit(ax, x, result, color, lw=1)
        pearsons.append(pearsonr(y, result.eval(x=x))[0])

print(f"mean Pearson R over {len(pearsons)} fitted configs: {np.mean(pearsons):.3f}")

ax.set_xscale("log")
ax.set_xlabel("UMI per cell")
ax.set_ylabel("protein MI (bits)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5)
utils.save(fig, "05_hparam_noise_scaling")
