import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

df = pd.read_csv(utils.path("hyperpam_sweep.csv"))
# The default (main-text) configuration is logged without hyperparameter values;
# label it "default" rather than dropping it.
df["trial_id"] = df["trial_id"].fillna(-1)
df = utils.add_umis(df, "PBMC")


def r_squared(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


# Fit every configuration and record its R^2.
configs = []
for t, sub in df.groupby("trial_id"):
    sub = sub.sort_values("umis_per_cell")
    x, y = sub["umis_per_cell"].values, sub["mi"].values
    result = utils.fit_noise(x, y)
    r2 = r_squared(y, result.eval(x=x))
    c = sub.iloc[0]
    if t == -1:
        label = f"default (R$^2$={r2:.2f})"
    else:
        label = (
            f"lr={c.max_lr:.0e} bs={int(c.batch_size)} "
            f"do={c.dropout:.1f} wd={c.weight_decay:.0e} (R$^2$={r2:.2f})"
        )
    configs.append((r2, x, y, result, label))

# Order the legend by descending R^2.
configs.sort(key=lambda v: -v[0])

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for i, (r2, x, y, result, label) in enumerate(configs):
    color = cmap(i % 10)
    ax.scatter(x, y, color=color, s=10, label=label)
    utils.plot_fit(ax, x, result, color, lw=1)

r2s = [c[0] for c in configs]
print(f"mean R^2 over all {len(r2s)} configs: {np.mean(r2s):.3f} (median {np.median(r2s):.3f})")
print(f"mean R^2 excluding collapsed configs (R^2>0.1): {np.mean([r for r in r2s if r > 0.1]):.3f}")

ax.set_xscale("log")
ax.set_xlabel("UMI per cell")
ax.set_ylabel("protein MI (bits)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5)
utils.save(fig, "05_hparam_noise_scaling")
