"""Generate all figures for the STATE SE + ESM initial results report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "figures"

COLORS = {
    "PCA": "#4C72B0",
    "RandomProjection": "#DD8452",
    "SCVI": "#55A868",
    "Geneformer": "#C44E52",
    "State": "#8172B3",
}

# ── All MI data from disk ──
MI_DATA = {
    "PBMC": {
        "protein_counts": {
            "PCA":              {"mean": 3.8365, "std": 0.0507, "n": 4},
            "RandomProjection": {"mean": 2.8724, "std": 0.0041, "n": 4},
            "SCVI":             {"mean": 4.7443, "std": 0.0604, "n": 4},
            "Geneformer":       {"mean": 3.7726, "std": 0.0567, "n": 4},
            "State":            {"mean": 0.3780, "std": 0.0,    "n": 1},
        },
        "celltype.l3": {
            "PCA":              {"mean": 3.1798, "std": 0.0173, "n": 4},
            "RandomProjection": {"mean": 2.4130, "std": 0.0186, "n": 4},
            "SCVI":             {"mean": 3.6329, "std": 0.0300, "n": 4},
            "Geneformer":       {"mean": 3.1962, "std": 0.0209, "n": 4},
            "State":            {"mean": 0.4048, "std": 0.0,    "n": 1},
        },
    },
    "larry": {
        "clone": {
            "PCA":              {"mean": 1.4759, "std": 0.0464, "n": 3},
            "RandomProjection": {"mean": 0.4020, "std": 0.0,    "n": 3},
            "SCVI":             {"mean": 1.7257, "std": 0.0548, "n": 3},
            "Geneformer":       {"mean": 0.7708, "std": 0.1423, "n": 3},
            "State":            {"mean": 0.0153, "std": 0.0,    "n": 1},
        },
    },
    "merfish": {
        "cur_idx": {
            "PCA":              {"mean": 1.5666, "std": 0.0351, "n": 4},
            "RandomProjection": {"mean": 0.4880, "std": 0.0078, "n": 4},
            "SCVI":             {"mean": 1.9323, "std": 0.0282, "n": 4},
            "State":            {"mean": 1.7027, "std": 0.0,    "n": 1},
        },
        "ng_idx": {
            "PCA":              {"mean": 1.6224, "std": 0.0261, "n": 4},
            "RandomProjection": {"mean": 0.5049, "std": 0.0055, "n": 4},
            "SCVI":             {"mean": 1.8932, "std": 0.0072, "n": 4},
            "Geneformer":       {"mean": 1.0497, "std": 0.0658, "n": 4},
            "State":            {"mean": 1.6319, "std": 0.0,    "n": 1},
        },
    },
    "shendure": {
        "author_day": {
            "PCA":              {"mean": 1.1087, "std": 0.0, "n": 1},
            "RandomProjection": {"mean": 0.4737, "std": 0.0, "n": 1},
            "SCVI":             {"mean": 2.1670, "std": 0.0, "n": 1},
            "Geneformer":       {"mean": 1.7796, "std": 0.0, "n": 1},
            "State":            {"mean": 0.0743, "std": 0.0, "n": 1},
        },
    },
}


def make_barplot(dataset, signal, data, outpath):
    algos = list(data.keys())
    means = [data[a]["mean"] for a in algos]
    stds = [data[a]["std"] for a in algos]
    ns = [data[a]["n"] for a in algos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        algos, means, yerr=stds, capsize=4,
        color=[COLORS.get(a, "#999") for a in algos],
        edgecolor="black", linewidth=0.5,
    )

    for bar, mean, std, n in zip(bars, means, stds, ns):
        label = f"{mean:.3f}"
        if n > 1:
            label += f"\n$\\pm${std:.3f} (n={n})"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.02,
                label, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(f"LMI Mutual Information ({signal})")
    ax.set_title(f"{dataset} -- LMI comparison ({signal})")
    ax.grid(axis="y", alpha=0.3)
    ymax = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, ymax * 1.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def make_summary_barplot():
    """One multi-panel figure showing STATE vs best-other for all datasets."""
    datasets = ["PBMC", "larry", "merfish", "shendure"]
    signals = ["protein_counts", "clone", "ng_idx", "author_day"]
    labels = ["PBMC\n(protein_counts)", "Larry\n(clone)", "MERFISH\n(ng_idx)", "Shendure\n(author_day)"]

    state_vals = []
    best_other_vals = []
    best_other_names = []
    rp_vals = []

    for ds, sig in zip(datasets, signals):
        d = MI_DATA[ds][sig]
        state_vals.append(d["State"]["mean"])
        rp_vals.append(d["RandomProjection"]["mean"])
        best_name, best_val = None, -1
        for algo, v in d.items():
            if algo != "State" and v["mean"] > best_val:
                best_val = v["mean"]
                best_name = algo
        best_other_vals.append(best_val)
        best_other_names.append(best_name)

    x = np.arange(len(datasets))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, best_other_vals, w, label="Best other", color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(x, rp_vals, w, label="RandomProjection", color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, state_vals, w, label="State (ESM)", color="#8172B3", edgecolor="black", linewidth=0.5)

    for i in range(len(datasets)):
        ax.text(x[i] - w, best_other_vals[i] + 0.05, f"{best_other_vals[i]:.2f}\n({best_other_names[i]})",
                ha="center", va="bottom", fontsize=7)
        ax.text(x[i], rp_vals[i] + 0.05, f"{rp_vals[i]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + w, state_vals[i] + 0.05, f"{state_vals[i]:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LMI (nats)")
    ax.set_title("STATE SE (ESM) vs Other Methods -- All Datasets")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUTDIR}/summary_comparison.png")


def make_esm_coverage():
    """Bar chart of ESM gene coverage per dataset."""
    datasets = ["PBMC", "larry", "merfish", "shendure"]
    total = [20729, 25289, 483, 91013]
    mapped = [14899, 18146, 479, 40856]
    coverage = [100 * m / t for m, t in zip(mapped, total)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(datasets, coverage,
                  color=["#4C72B0", "#55A868", "#C44E52", "#DD8452"],
                  edgecolor="black", linewidth=0.5)
    for bar, cov, m, t in zip(bars, coverage, mapped, total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cov:.1f}%\n({m}/{t})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ESM Coverage (%)")
    ax.set_title("ESM2 Embedding Coverage per Dataset")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/esm_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUTDIR}/esm_coverage.png")


if __name__ == "__main__":
    print("Generating figures...")

    # Per-dataset barplots
    for ds, signals in MI_DATA.items():
        for sig, data in signals.items():
            make_barplot(ds, sig, data, f"{OUTDIR}/{ds}_{sig}_comparison.png")

    # Summary figure
    make_summary_barplot()

    # ESM coverage
    make_esm_coverage()

    print("Done.")
