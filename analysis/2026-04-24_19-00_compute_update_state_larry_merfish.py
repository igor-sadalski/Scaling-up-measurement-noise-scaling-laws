"""Append State MI values for larry and merfish to collect_mi_results.csv
and regenerate collect_mi_results.png.

The existing CSV already contains State rows for PBMC and shendure, plus all
five algorithms for PBMC/larry/merfish/shendure. It was missing State rows
for larry (signal=clone) and merfish (signal=ng_idx). This script scans
<DATA_ROOT>/{larry,merfish}/<size>/<quality>/results/State/model/MI/<seed>/
Y_<signal>_<quality>/lmi_mutual_information.txt, reads each value, and
appends the new rows to collect_mi_results.csv preserving the existing
schema (dataset,size,quality,algorithm,signal,seed,mi_value,umis_per_cell).
The PNG is regenerated from the updated CSV using the same plotting logic
as 2026-04-14_15-25_collect_and_plot_mi_scaling.ipynb.
"""

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


_log_f = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
print(f"Logging to {LOG_PATH}")

import os
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.auto import tqdm


DATA_ROOT = Path("/home/igor/noise_scaling/data")
FINAL_DIR = Path(
    "/home/igor/noise_scaling/Scaling-up-measurement-noise-scaling-laws/analysis/final_results"
)
CSV_PATH = FINAL_DIR / "collect_mi_results.csv"
PNG_PATH = FINAL_DIR / "collect_mi_results.png"


# Same config as the plotting notebook. State larry/merfish emit seed=42
# (larry) and seeds=[42,1404,2303] (merfish); we accept whichever are found.
SEEDS = [42, 1404, 2303, 2701]
EXPECTED_NEW = {
    "larry": {
        "sizes": [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        "qualities": [
            0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311,
            0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0,
        ],
        "signals": ["clone"],
    },
    "merfish": {
        "sizes": [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        "qualities": [
            0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267,
            0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0,
        ],
        "signals": ["ng_idx"],
    },
}


def _read_mi(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return np.nan


def scan_state_mi():
    rows = []
    for ds, cfg in EXPECTED_NEW.items():
        for sz, q, sig, sd in product(cfg["sizes"], cfg["qualities"], cfg["signals"], SEEDS):
            stem = f"Y_{sig}_{q}"
            mi_path = (
                DATA_ROOT / ds / str(sz) / str(q) / "results" / "State" /
                "model" / "MI" / str(sd) / stem / "lmi_mutual_information.txt"
            )
            rows.append({
                "dataset": ds, "size": sz, "quality": q,
                "algorithm": "State", "signal": sig, "seed": sd,
                "path": str(mi_path),
            })

    df = pd.DataFrame(rows)
    print(f"Total expected State larry+merfish: {len(df)}")

    with ThreadPoolExecutor(max_workers=128) as pool:
        df["exists"] = list(tqdm(pool.map(os.path.exists, df["path"]),
                                 total=len(df), desc="Scanning"))
    found = df[df["exists"]].copy()
    print(f"Found: {len(found)}, Missing: {len(df) - len(found)}")

    with ThreadPoolExecutor(max_workers=128) as pool:
        found["mi_value"] = list(tqdm(pool.map(_read_mi, found["path"]),
                                      total=len(found), desc="Reading MI"))
    return found


def umis_per_cell_map(existing_df):
    # existing CSV already contains the quality -> umis_per_cell mapping
    # (quality fully determines umis_per_cell within a dataset).
    mapping = {}
    for ds in EXPECTED_NEW:
        sub = existing_df[existing_df["dataset"] == ds][["quality", "umis_per_cell"]]
        mapping[ds] = dict(sub.drop_duplicates().values)
    return mapping


def _fmt_size(sz):
    if sz >= 1_000_000:
        return f"{sz/1_000_000:.0f}M" if sz % 1_000_000 == 0 else f"{sz/1_000_000:.1f}M"
    if sz >= 1_000:
        return f"{sz/1_000:.0f}k" if sz % 1_000 == 0 else f"{sz/1_000:.1f}k"
    return str(sz)


ALGO_ORDER = ["PCA", "RandomProjection", "SCVI", "Geneformer", "State"]


def plot_mi_vs_quality(df):
    # One row per (dataset, signal). Columns = algorithms. Legend = cell count.
    all_ds_sig = sorted(set(zip(df["dataset"], df["signal"])))
    algorithms = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    n_rows = len(all_ds_sig)
    n_cols = len(algorithms)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols + 1.6, 3.5 * n_rows + 0.8),
        squeeze=False, sharex=False,
    )
    for i in range(n_rows):
        for j in range(1, n_cols):
            axes[i, j].sharey(axes[i, 0])

    cmap = plt.cm.viridis
    ds_norms = {}
    for ds, _ in all_ds_sig:
        sizes = sorted(df[df["dataset"] == ds]["size"].unique())
        if sizes:
            ds_norms[ds] = mcolors.LogNorm(vmin=min(sizes), vmax=max(sizes))

    for i, (ds, sig) in enumerate(all_ds_sig):
        norm = ds_norms.get(ds)
        for j, algo in enumerate(algorithms):
            ax = axes[i, j]
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo) & (df["signal"] == sig)]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)
                ax.set_xscale("log")
            else:
                for sz in sorted(sub["size"].unique()):
                    color = cmap(norm(sz))
                    sz_df = sub[sub["size"] == sz]
                    agg = (sz_df.groupby("quality")["mi_value"]
                           .agg(["mean", "std", "count"]).reset_index()
                           .sort_values("quality"))
                    agg["sem"] = (agg["std"] / np.sqrt(agg["count"])).fillna(0)
                    ax.errorbar(agg["quality"], agg["mean"], yerr=agg["sem"],
                                marker="o", markersize=3, linewidth=1, capsize=2,
                                color=color, alpha=0.85, label=_fmt_size(sz))
                ax.set_xscale("log")

            if i == n_rows - 1:
                ax.set_xlabel("Quality (downsampling ratio)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ds} — MI ({sig})", fontsize=10)
            if i == 0:
                ax.set_title(algo, fontsize=12, fontweight="bold")
            ax.tick_params(labelsize=8)
            if j > 0:
                ax.tick_params(labelleft=False)

        last_ax = axes[i, n_cols - 1]
        handles, labels = last_ax.get_legend_handles_labels()
        if handles:
            last_ax.legend(handles, labels, title="# cells", fontsize=6,
                           title_fontsize=7, loc="center left",
                           bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
                           frameon=True, framealpha=0.8)

    fig.suptitle("MI vs Quality (from disk)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 0.92, 0.96])
    return fig


def main():
    print(f"Reading existing CSV: {CSV_PATH}")
    df_existing = pd.read_csv(CSV_PATH)
    print(f"Existing rows: {len(df_existing)}")

    # Drop any pre-existing State larry/merfish rows so the update is idempotent.
    mask_drop = (
        (df_existing["algorithm"] == "State")
        & df_existing["dataset"].isin(list(EXPECTED_NEW))
    )
    if mask_drop.any():
        print(f"Dropping {mask_drop.sum()} stale State rows for larry/merfish")
    df_keep = df_existing[~mask_drop].copy()

    found = scan_state_mi()
    q2u = umis_per_cell_map(df_existing)

    new_rows = found.copy()
    new_rows["umis_per_cell"] = [
        q2u[ds][q] for ds, q in zip(new_rows["dataset"], new_rows["quality"])
    ]
    out_cols = ["dataset", "size", "quality", "algorithm",
                "signal", "seed", "mi_value", "umis_per_cell"]
    new_rows = new_rows[out_cols]
    print(f"New State larry+merfish rows: {len(new_rows)}")

    df_combined = pd.concat([df_keep, new_rows], ignore_index=True)
    df_combined = df_combined.sort_values(
        ["dataset", "size", "quality", "algorithm", "signal", "seed"]
    ).reset_index(drop=True)

    print(f"Writing updated CSV: {CSV_PATH} ({len(df_combined)} rows)")
    df_combined.to_csv(CSV_PATH, index=False)

    print("Regenerating plot (excluding celltype.l3)")
    df_plot = df_combined[df_combined["signal"] != "celltype.l3"]
    fig = plot_mi_vs_quality(df_plot)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {PNG_PATH}")

    # Summary
    print("\n--- State rows per dataset (final) ---")
    state = df_combined[df_combined["algorithm"] == "State"]
    print(state.groupby("dataset").size().to_string())


if __name__ == "__main__":
    main()
