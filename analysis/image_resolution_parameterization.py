"""Empirical justification of the SNR for resolution downsampling (Reviewer 1.6).

The manuscript parameterizes the SNR as eta = 1/f for an *area* downsampling
factor f (a tile of sqrt(f) x sqrt(f) pixels is replaced by its mean). In the
codebase this is avg_pool2d with kernel = sqrt(f), so f = kernel**2.

Instead of asserting eta = 1/f, we measure the SNR empirically on TissueMNIST-224
(matching the experiment's degradation x' = Resize_224(avgpool(x, sqrt(f)))) as the
variance-ratio SNR  eta_emp = Var(x) / E[(x-x')^2]  (analogous to the single-cell
eta = CV^-2), and check (a) how eta_emp scales with 1/f and (b) that the universal
noise-scaling law fits the label MI in this empirical SNR.

Run in the `lt` env: conda activate lt && python analysis/image_resolution_parameterization.py
"""
import os, re, glob
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from lmfit import Model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PIX_GLOB = os.path.join(REPO_ROOT, "images", "tissuemnist_models", "result_pix_*.csv")
NPZ_PATH = os.path.expanduser("~/.medmnist/tissuemnist_224.npz")
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
OUT_PNG = os.path.join(FIGDIR, "image_resolution_parameterization.png")
OUT_PDF = os.path.join(FIGDIR, "image_resolution_parameterization.pdf")
OUT_A_PNG = os.path.join(FIGDIR, "image_resolution_vs_emp_snr.png")
OUT_A_PDF = os.path.join(FIGDIR, "image_resolution_vs_emp_snr.pdf")
SNR_CSV = os.path.join(HERE, "final_results", "image_resolution_snr.csv")

N_SAMPLE, IMG_SIZE, SEED = 4000, 224, 0
KERNELS = [2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224]   # avg_pool2d kernel = sqrt(f_manuscript)

# ---- big_fig_2 styling ----
sns.set_style("whitegrid")
rcParams.update({"figure.dpi": 150, "grid.linewidth": 0.5, "grid.linestyle": "dashed",
                 "legend.fancybox": False, "mathtext.fontset": "stix"})
mpl.rcParams["pdf.fonttype"] = 42; mpl.rcParams["svg.fonttype"] = "none"
c6 = ["#c4bae2", "#40ada6", "#3c4ebf", "#e3a600", "#d6543a", "#edb1a3"]
PALETTE_TISSUE = ("#CBC106FF, #27993CFF, #1C6838FF, #8EBCB5FF, #389CA7FF, "
                  "#4D83ABFF, #CB7B26FF, #BF565DFF, #9E163CFF").split(", ")
LABEL_MAP = {f"ova_mi_continuous_Class_{i}": n for i, n in enumerate(
    ["Collecting Duct, Connecting Tubule", "Distal Convoluted Tubule",
     "Glomerular endothelial cells", "Interstitial endothelial cells", "Leukocytes",
     "Podocytes", "Proximal Tubule Segments", "Thick Ascending Limb"])}
LABEL_MAP["mi_score"] = "8-class MI"


def info_scaling_model(x, A, B):
    return 0.5 * np.log2((x * B + 1) / (1 + A * x))


def fit_info(x, y):
    m = Model(info_scaling_model)
    p = m.make_params(A=1e-2, B=1e-2); p["A"].min = p["B"].min = 0
    return m.fit(np.asarray(y, float), p, x=np.asarray(x, float))


def degrade(x, k):
    return F.interpolate(F.avg_pool2d(x, k, k), size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)


def measure_snr():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"{NPZ_PATH} not found; download tissuemnist_224.npz from "
            "https://zenodo.org/records/10519652/files/tissuemnist_224.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    imgs = np.load(NPZ_PATH, mmap_mode="r")["test_images"]
    idx = np.sort(rng.choice(imgs.shape[0], min(N_SAMPLE, imgs.shape[0]), replace=False))
    x_all = torch.from_numpy(imgs[idx].astype(np.float32) / 255.0).unsqueeze(1)
    var = float(x_all.var().item())
    rows = []
    for k in KERNELS:
        se, n = 0.0, 0
        for b in range(0, x_all.shape[0], 256):
            x = x_all[b:b + 256].to(dev)
            se += float(((x - degrade(x, k)) ** 2).sum().item()); n += x.numel()
        mse = se / n
        rows.append({"kernel": k, "f": k * k, "inv_f": 1.0 / (k * k),
                     "mse": mse, "eta_emp": var / mse})
    df = pd.DataFrame(rows); df.attrs["var"] = var
    return df


def load_mi():
    parts = []
    for fp in glob.glob(PIX_GLOB):
        d = pd.read_csv(fp); d["kernel"] = int(re.search(r"pix_(.+)x", fp).group(1))
        parts.append(d)
    pix = pd.concat(parts, ignore_index=True)
    return pix[pix["kernel"].isin(KERNELS)].sort_values("kernel")


def main():
    snr = measure_snr()
    eta_of_k = dict(zip(snr["kernel"], snr["eta_emp"]))
    inv_f, eta = snr["inv_f"].values, snr["eta_emp"].values

    cprop = np.sum(eta * inv_f) / np.sum(inv_f ** 2)         # proportional fit eta_emp = cprop * (1/f)
    r2 = 1 - np.sum((eta - cprop * inv_f) ** 2) / np.sum((eta - eta.mean()) ** 2)
    print(f"Var(x)={snr.attrs['var']:.5f}")
    print(snr[["kernel", "f", "inv_f", "mse", "eta_emp"]].to_string(index=False))
    print(f"\neta_emp = {cprop:.1f} * (1/f)  (proportional fit, manuscript area factor f),  R^2 = {r2:.3f}")

    pix = load_mi()
    cols = ["mi_score"] + [c for c in pix.columns if "ova_mi_continuous" in c]
    fits, r2s = {}, []
    print(f"\n{'MI column':>28} {'R^2(eta_emp)':>12}")
    for c in cols:
        sub = pix[["kernel", c]].dropna()
        x = np.array([eta_of_k[k] for k in sub["kernel"]]); y = sub[c].values
        res = fit_info(x, y); fits[c] = (x, y, res); r2s.append(res.rsquared)
        print(f"{c:>28} {res.rsquared:>12.3f}")
    print(f"{'MEAN':>28} {np.mean(r2s):>12.3f}")

    os.makedirs(os.path.dirname(SNR_CSV), exist_ok=True)
    snr.to_csv(SNR_CSV, index=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # a) empirical SNR vs 1/f (manuscript area factor)
    axA.scatter(inv_f, eta, color=c6[2], s=40, zorder=3, label="measured")
    xs = np.array([inv_f.min(), inv_f.max()])
    axA.plot(xs, cprop * xs, "--", color="grey", alpha=0.8,
             label=fr"$\eta_{{\rm emp}}\propto 1/f$  ($R^2={r2:.3f}$)")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"$1/f$  (inverse area downsampling factor)", fontsize=12)
    axA.set_ylabel(r"$\eta_{\rm emp} = \mathrm{Var}(x)\,/\,\mathbb{E}[(x-x')^2]$", fontsize=12)
    axA.legend(fontsize=9, loc="upper left")
    axA.text(-0.18, 1.03, "a)", transform=axA.transAxes, fontsize=16, fontweight="bold")

    # b) scaling law holds in the empirical SNR
    for i, c in enumerate(cols[::-1]):
        x, y, res = fits[c]; color = PALETTE_TISSUE[i % len(PALETTE_TISSUE)]
        xf = np.logspace(np.log10(x.min() / 5), np.log10(x.max() * 5), 2000)
        yf, ye = res.eval(x=xf), res.eval_uncertainty(x=xf, sigma=2)
        axB.fill_between(xf, yf + ye, yf - ye, color=color, alpha=0.2)
        axB.plot(xf, yf, "--", color=color, alpha=0.5)
        axB.scatter(x, y, color=color, s=30, label=LABEL_MAP.get(c, c))
    axB.set_xscale("log")
    axB.set_xlabel(r"empirical SNR  $\eta_{\rm emp}$", fontsize=12)
    axB.set_ylabel(r"label MI  $I(L; z)$ (bits)", fontsize=12)
    axB.legend(loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=8)
    axB.text(-0.18, 1.03, "b)", transform=axB.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight"); fig.savefig(OUT_PDF, bbox_inches="tight")

    # standalone panel a (no panel label), empirical SNR vs 1/f
    figA, axA1 = plt.subplots(1, 1, figsize=(5.5, 4.2), dpi=150)
    axA1.scatter(inv_f, eta, color=c6[2], s=40, zorder=3, label="measured")
    axA1.plot(xs, cprop * xs, "--", color="grey", alpha=0.8,
              label=fr"$\eta_{{\rm emp}}\propto 1/f$  ($R^2={r2:.3f}$)")
    axA1.set_xscale("log"); axA1.set_yscale("log")
    axA1.set_xlabel(r"$1/f$  (inverse area downsampling factor)", fontsize=12)
    axA1.set_ylabel(r"$\eta_{\rm emp} = \mathrm{Var}(x)\,/\,\mathbb{E}[(x-x')^2]$", fontsize=12)
    axA1.legend(fontsize=9, loc="upper left")
    figA.tight_layout()
    figA.savefig(OUT_A_PNG, dpi=150, bbox_inches="tight"); figA.savefig(OUT_A_PDF, bbox_inches="tight")

    print(f"\nSaved -> {os.path.relpath(SNR_CSV, REPO_ROOT)}, {os.path.relpath(OUT_PNG, REPO_ROOT)}, "
          f"{os.path.relpath(OUT_A_PNG, REPO_ROOT)}")


if __name__ == "__main__":
    main()
