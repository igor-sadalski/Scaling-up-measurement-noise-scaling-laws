import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

df = pd.read_csv(utils.path("geneformer_model_size_sweep.csv"))
df = utils.add_umis(df, "PBMC")

sizes = sorted(df["trainable_params_M"].unique())
cmap = plt.get_cmap("viridis")
colors = {p: cmap(i / max(1, len(sizes) - 1)) for i, p in enumerate(sizes)}

fig, ax = plt.subplots(figsize=(3.2, 2.6))
for p in sizes:
    sub = df[df["trainable_params_M"] == p].sort_values("umis_per_cell")
    x, y = sub["umis_per_cell"].values, sub["mi_protein_counts"].values
    ax.scatter(x, y, color=colors[p], s=10, label=f"{p:.1f}M")
    if len(x) >= 4:
        utils.plot_fit(ax, x, utils.fit_noise(x, y), colors[p], lw=1)

ax.set_xscale("log")
ax.set_xlabel("UMI per cell")
ax.set_ylabel("protein MI (bits)")
ax.legend(title="model size", bbox_to_anchor=(1.02, 1), loc="upper left")
utils.save(fig, "01_model_size_noise")
