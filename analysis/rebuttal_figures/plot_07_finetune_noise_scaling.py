import pandas as pd
import matplotlib.pyplot as plt
import utils

SIZE = 100000

ft = pd.read_csv(utils.path("finetune_se_100m_state_pbmc_noise_scaling.csv"))
ft = utils.add_umis(ft[ft["size"] == SIZE], "PBMC").sort_values("umis_per_cell")

sc = utils.load_collect()
sc = sc[(sc["dataset"] == "PBMC") & (sc["algorithm"] == "State") &
        (sc["signal"] == "protein_counts") & (sc["size"] == SIZE)]
sc = sc.groupby("umis_per_cell")["mi_value"].mean().reset_index().sort_values("umis_per_cell")

fig, ax = plt.subplots(figsize=(3.2, 2.6))
for x, y, color, label in [
    (ft["umis_per_cell"].values, ft["mi"].values, utils.c6[4], "fine-tuned SE-100M"),
    (sc["umis_per_cell"].values, sc["mi_value"].values, utils.c6[2], "from-scratch STATE"),
]:
    ax.scatter(x, y, color=color, s=12, label=label)
    utils.plot_fit(ax, x, utils.fit_noise(x, y), color, lw=1)

ax.set_xscale("log")
ax.set_xlabel("UMI per cell")
ax.set_ylabel("protein MI (bits)")
ax.legend(loc="upper left", fontsize=6)
utils.save(fig, "07_finetune_noise_scaling")
