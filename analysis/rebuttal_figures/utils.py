import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib as mpl
from lmfit import Model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "final_results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

sns.set_style("whitegrid")
rcParams.update({
    "figure.dpi": 150,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "grid.linewidth": 0.5,
    "grid.linestyle": "dashed",
    "legend.fancybox": False,
    "mathtext.fontset": "stix",
})
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

c6 = ["#c4bae2", "#40ada6", "#3c4ebf", "#e3a600", "#d6543a", "#edb1a3"]
METHOD_ORDER = ["RandomProjection", "PCA", "SCVI", "Geneformer", "State"]
METHOD_LABEL = {
    "RandomProjection": "rand. proj.",
    "PCA": "PCA",
    "SCVI": "scVI",
    "Geneformer": "Geneformer",
    "State": "STATE",
}
METHOD_COLOR = {
    "RandomProjection": c6[1],
    "PCA": c6[2],
    "SCVI": c6[3],
    "Geneformer": c6[4],
    "State": c6[0],
}
DATASET_SIGNAL = {
    "PBMC": "protein_counts",
    "larry": "clone",
    "merfish": "ng_idx",
    "shendure": "author_day",
}


def path(name):
    return os.path.join(DATA, name)


def load_collect():
    return pd.read_csv(path("collect_mi_results.csv"))


def umi_map(dataset):
    df = load_collect()
    df = df[df["dataset"] == dataset][["quality", "umis_per_cell"]].drop_duplicates()
    return {round(q, 6): u for q, u in zip(df["quality"], df["umis_per_cell"])}


def add_umis(df, dataset):
    m = umi_map(dataset)
    out = df.copy()
    out["umis_per_cell"] = out["quality"].round(6).map(m)
    return out


def info_scaling_model(x, A, B):
    return 0.5 * np.log2((x * B + 1) / (1 + A * x))


def fit_noise(x, y):
    model = Model(info_scaling_model)
    params = model.make_params(A=1e-2, B=1e-2)
    params["A"].min = params["B"].min = 0
    return model.fit(np.asarray(y), params, x=np.asarray(x))


def plot_fit(ax, x, result, color, **kw):
    xf = np.logspace(np.log10(min(x) / 5), np.log10(max(x) * 5), 500)
    ax.plot(xf, result.eval(x=xf), linestyle="--", color=color, **kw)


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
