import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import kendalltau
import utils

ksg = pd.read_csv(utils.path("ksg_vs_quality_shendure_SCVI.csv"))
ksg["q"] = ksg["quality"].round(6)

mi = utils.load_collect()
mi = mi[(mi["dataset"] == "shendure") & (mi["algorithm"] == "SCVI") & (mi["signal"] == "author_day")]
mi = mi.groupby(["size", "quality", "umis_per_cell"])["mi_value"].mean().reset_index()
mi["q"] = mi["quality"].round(6)

df = mi.merge(ksg[["size", "q", "mi_bits"]], on=["size", "q"])

tau, p = kendalltau(df["mi_value"], df["mi_bits"])
print(f"Kendall tau (LMI vs KSG) = {tau:.3f} (p={p:.1e}, n={len(df)})")

fig, ax = plt.subplots(figsize=(3.2, 2.6))
sc = ax.scatter(df["mi_value"], df["mi_bits"], c=df["umis_per_cell"],
                cmap="viridis", norm=mcolors.LogNorm(), s=12,
                edgecolor="k", linewidth=0.2)
lims = [0, max(df["mi_value"].max(), df["mi_bits"].max()) * 1.05]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("LMI (bits)")
ax.set_ylabel("KSG MI (bits)")
ax.text(0.05, 0.95, rf"Kendall $\tau$ = {tau:.2f}", transform=ax.transAxes,
        ha="left", va="top", fontsize=7)
fig.colorbar(sc, ax=ax, label="UMI per cell")
utils.save(fig, "04_shendure_lmi_vs_ksg")
