import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import utils

df = pd.read_csv(utils.path("geneformer_loss_curves_sml.csv"))
df = utils.add_umis(df, "PBMC")

curves = df[(df["dataset"] == "PBMC") & (df["size"] == 4641) & (df["curve"] == "val")]
umis = sorted(curves["umis_per_cell"].unique())
cmap = plt.get_cmap("viridis")
norm = mcolors.LogNorm(vmin=min(umis), vmax=max(umis))

fig, ax = plt.subplots(figsize=(3.2, 2.6))
for u in umis:
    sub = curves[curves["umis_per_cell"] == u].sort_values("step")
    ax.plot(sub["step"], sub["loss"], marker="o", ms=2, lw=1, color=cmap(norm(u)))
ax.set_xlabel("training steps")
ax.set_ylabel("validation loss")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="UMI per cell")
utils.save(fig, "06a_geneformer_loss_curves")

best = df[(df["dataset"] == "PBMC") & df["is_best_checkpoint"]]
sizes = sorted(best["size"].unique())
scmap = plt.get_cmap("viridis")
snorm = mcolors.LogNorm(vmin=min(sizes), vmax=max(sizes))

fig, ax = plt.subplots(figsize=(3.2, 2.6))
for s in sizes:
    sub = best[best["size"] == s].sort_values("umis_per_cell")
    ax.plot(sub["umis_per_cell"], sub["step"], marker="o", ms=3, lw=1,
            color=scmap(snorm(s)), label=f"{s:,}")
ax.set_xscale("log")
ax.set_xlabel("UMI per cell")
ax.set_ylabel("steps to optimal model")
ax.legend(title="cell number", bbox_to_anchor=(1.02, 1), loc="upper left")
utils.save(fig, "06b_geneformer_best_step")
