#!/mnt/home/gokulg/miniconda3/envs/lt/bin/python
#SBATCH --job-name=batch_effect_pilot
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.err
#
# batch_effect_experiment.py
#
# Directly submittable:  `sbatch batch_effect_experiment.py`
# (the shebang points at the `lt` conda env, which has torch + torchvision + medmnist +
#  latentmi, so no `conda activate` is needed; the #SBATCH lines above are plain comments
#  to python). You can still run it by hand: `python batch_effect_experiment.py ...`.
#
# Batch-effect extension of the tissuemnist image-noise MI experiment.
#
# It reuses the *exact* conventions of ../run_kidney_experiment.py:
#   - dataset:            tissuemnist via medmnist, loaded at native `size` (default 224)
#   - image pipeline:     ToTensor -> repeat grayscale to 3 channels -> add Gaussian
#                         measurement noise (randn * noise_level) -> Resize(size)
#   - classifier:         mobilenet_v3_small (IMAGENET1K_V1), final Linear -> num_classes
#   - training:           Adam(lr=1e-3), CrossEntropyLoss
#   - MI estimator:       ksg.midd(Y_true, Y_pred)  (discrete MI between predicted and
#                         true labels, in bits) -- the "existing estimator" for label MI.
#
# It ADDS a simulated batch effect (an additive, uncorrected per-batch pixel shift) and
# runs a 2-D sweep over (measurement noise, batch strength). See README.md for the theory.
#
# Parameterization / bookkeeping (see the module docstring of ../run_kidney_experiment.py):
#   noise_level == sigma passed to add_gauss(x, sigma) == per-pixel measurement-noise std.
#   => sigma_meas^2 = noise_level^2   (variance of the per-image measurement noise / pixel)
#   We define the SNR knob as   eta = 1 / sigma_meas^2   (signal variance is a fixed
#   constant of the dataset, so it only rescales eta by a constant and cancels out of the
#   collapse test eta/eta_eff = 1 + sigma_M^2/sigma_meas^2).
#
# The batch shift mu_b ~ N(0, sigma_M^2 I) has the same shape as the (3-channel) image and
# is added AFTER the per-image measurement noise and BEFORE the (identity) Resize, i.e. at
# the same native resolution and in the same units as sigma_meas -- so sigma_M^2/sigma_meas^2
# is a well-defined per-pixel ratio.

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import medmnist
from medmnist import INFO
from latentmi import ksg
from tqdm.auto import tqdm

# When submitted with `sbatch batch_effect_experiment.py`, Slurm copies this file to a
# spool dir (/var/spool/slurm/.../slurm_script), so __file__ no longer sits next to the
# repo. Pin the real locations by absolute path (fall back to __file__ for portability),
# so both `from run_kidney_experiment import ...` and `import analyze_batch_effect` resolve.
PILOT_DIR = Path('/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot')
if not PILOT_DIR.exists():
    PILOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PILOT_DIR.parent
for _d in (str(PILOT_DIR), str(IMAGES_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Reuse the model + training loop verbatim from the sibling experiment so the classifier
# and optimizer are identical to the no-batch scaling curve.
from run_kidney_experiment import create_model, train_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger('batch_effect')


# --------------------------------------------------------------------------------------
# Batch-effect dataset
# --------------------------------------------------------------------------------------
class BatchEffectDataset(Dataset):
    """Wrap a medmnist dataset and apply, per item:

        ToTensor -> repeat-to-3ch -> + measurement noise (N(0, sigma_meas^2))
                 -> + batch shift mu_{b(i)} -> Resize(size)

    The measurement noise is made deterministic per (seed, base-index) so the whole
    experiment is reproducible for a given seed (the original applies fresh randn each
    epoch; for a variance-focused pilot a fixed realization is cleaner and does not affect
    the theory). Batch assignment and the shift vectors are supplied by the caller so that
    train and test share the same mu_b for a given (seed, sigma_M).
    """

    def __init__(self, base_ds, active_indices, batch_of_item, batch_shifts,
                 sigma_meas, size, noise_seed):
        self.base = base_ds
        self.active = np.asarray(active_indices)
        self.batch_of_item = np.asarray(batch_of_item)      # batch id per *item* (0..k-1)
        self.batch_shifts = batch_shifts                    # tensor [k, C, H, W] or None
        self.sigma_meas = float(sigma_meas)
        self.noise_seed = int(noise_seed)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size))

    def __len__(self):
        return len(self.active)

    def __getitem__(self, item):
        base_idx = int(self.active[item])
        img, label = self.base[base_idx]

        x = self.to_tensor(img)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        # per-image measurement noise (reproducible per seed+base index)
        if self.sigma_meas > 0:
            g = torch.Generator().manual_seed((self.noise_seed * 2654435761 + base_idx) % (2 ** 31))
            x = x + torch.randn(x.shape, generator=g) * self.sigma_meas

        # uncorrected additive per-batch shift
        if self.batch_shifts is not None:
            x = x + self.batch_shifts[int(self.batch_of_item[item])]

        x = self.resize(x)
        return x, label


def assign_batches(rng, n, k):
    """Assign n items uniformly at random to k batches; return int array of batch ids."""
    return rng.integers(0, k, size=n)


def make_batch_shifts(seed, k, shape, sigma_M):
    """mu_b = sigma_M * base_b, base_b ~ N(0, I). Fixing base per seed makes sigma_M=0 the
    exact baseline and larger sigma_M a scaled version of the same directions, which
    isolates the *magnitude* effect the theory is about."""
    if sigma_M == 0:
        return None
    rng = np.random.default_rng(seed + 777)  # shift RNG independent of the split RNG
    base = rng.standard_normal(size=(k, *shape)).astype(np.float32)
    return torch.from_numpy(base * float(sigma_M))


# --------------------------------------------------------------------------------------
# One (seed, noise_level, sigma_M) cell
# --------------------------------------------------------------------------------------
def evaluate_label_mi(model, loader, device):
    """MI between predicted and true labels via the existing discrete estimator (bits)."""
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='eval', leave=False):
            x = x.to(device)
            logits = model(x)
            yhats.append(logits.argmax(1).cpu().numpy())
            ys.append(y.squeeze().long().numpy())
    Y = np.concatenate(ys)
    Yhat = np.concatenate(yhats)
    return float(ksg.midd(Y, Yhat))


def run_cell(base_train, base_test, active_train, active_test, img_shape,
             seed, noise_level, sigma_M, num_classes, device, args):
    """Train + evaluate for one sweep cell; returns MI (bits)."""
    k = args.k
    # reproducible split (same structure for a given seed, independent of noise/sigma_M)
    split_rng = np.random.default_rng(seed)
    batch_of_train = assign_batches(split_rng, len(active_train), k)
    batch_of_test = assign_batches(split_rng, len(active_test), k)

    shifts = make_batch_shifts(seed, k, img_shape, sigma_M)

    train_ds = BatchEffectDataset(base_train, active_train, batch_of_train, shifts,
                                  noise_level, args.size, noise_seed=seed)
    test_ds = BatchEffectDataset(base_test, active_test, batch_of_test, shifts,
                                 noise_level, args.size, noise_seed=seed + 1)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    # deterministic model init + training order per cell
    torch.manual_seed(seed * 100003 + hash((round(noise_level, 6), round(sigma_M, 6))) % 100003)
    model = create_model(num_classes, device)

    log.info("training: %d train imgs, %d epochs (noise_level=%.4f, sigma_M=%.4f)",
             len(train_ds), args.epochs, noise_level, sigma_M)
    t0 = time.time()
    train_model(model, train_dl, None, device, n_epochs=args.epochs, lr=args.lr)
    log.info("training done in %.1fs; evaluating MI on %d test imgs", time.time() - t0, len(test_ds))

    mi = evaluate_label_mi(model, test_dl, device)
    return mi


# --------------------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------------------
def build_grids(args):
    noise_grid = np.array(args.noise_grid, dtype=float)
    # geometric middle of the measurement-noise grid -> where the ratio targets are set
    sigma_meas_mid = float(np.exp(np.mean(np.log(noise_grid))))
    ratio_targets = np.array(args.ratio_targets, dtype=float)      # sigma_M^2/sigma_meas^2 @ mid
    sigma_M_grid = sigma_meas_mid * np.sqrt(ratio_targets)
    return noise_grid, sigma_M_grid, sigma_meas_mid, ratio_targets


def main():
    p = argparse.ArgumentParser(description="Batch-effect MI sweep (tissuemnist).")
    p.add_argument('--dataset', default='tissuemnist')
    p.add_argument('--output-dir', default=str(PILOT_DIR / 'results'))
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--k', type=int, default=100, help='number of batches')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--noise-grid', type=float, nargs='+',
                   default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
                   help='measurement-noise std grid (== noise_level == sigma_meas)')
    p.add_argument('--ratio-targets', type=float, nargs='+',
                   default=[0.0, 0.25, 1.0, 4.0, 9.0, 16.0],
                   help='sigma_M^2/sigma_meas^2 targets at the middle of the noise grid')
    p.add_argument('--subset-train', type=int, default=1000,
                   help='subsample this many train images (for a fast pilot)')
    p.add_argument('--subset-test', type=int, default=1000,
                   help='subsample this many test images (for a fast pilot)')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--no-analyze', action='store_true',
                   help='skip building figures/summary.json at the end')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("device=%s | torch=%s | cuda_available=%s", device, torch.__version__, torch.cuda.is_available())
    log.info("args: %s", vars(args))

    info = INFO[args.dataset]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # raw datasets (transform=None -> returns PIL image, label)
    log.info("loading %s (size=%d) ...", args.dataset, args.size)
    base_train = DataClass(split='train', transform=None,
                           download=not args.no_download, size=args.size)
    base_test = DataClass(split='test', transform=None,
                          download=not args.no_download, size=args.size)
    log.info("full split: %d train, %d test, %d classes", len(base_train), len(base_test), num_classes)

    # image tensor shape after ToTensor + repeat-to-3ch (for building mu_b)
    x0 = transforms.ToTensor()(base_train[0][0])
    if x0.shape[0] == 1:
        x0 = x0.repeat(3, 1, 1)
    img_shape = tuple(x0.shape)  # (C, H, W)
    log.info("per-image tensor shape (C,H,W): %s", img_shape)

    noise_grid, sigma_M_grid, sigma_meas_mid, ratio_targets = build_grids(args)
    log.info("noise_grid (sigma_meas): %s", np.round(noise_grid, 4).tolist())
    log.info("sigma_meas_mid (geom): %.4f", sigma_meas_mid)
    log.info("sigma_M_grid: %s  (ratio targets @ mid: %s)",
             np.round(sigma_M_grid, 4).tolist(), ratio_targets.tolist())

    n_cells = len(args.seeds) * len(noise_grid) * len(sigma_M_grid)
    log.info("total sweep cells: %d (= %d seeds x %d noise x %d sigma_M)",
             n_cells, len(args.seeds), len(noise_grid), len(sigma_M_grid))

    rows = []
    results_csv = os.path.join(args.output_dir, 'results.csv')
    cell_bar = tqdm(total=n_cells, desc='sweep cells', unit='cell')
    for seed in args.seeds:
        # active index subset is fixed per seed (same images across the whole grid)
        sub_rng = np.random.default_rng(seed + 12345)
        if args.subset_train is not None and args.subset_train < len(base_train):
            active_train = np.sort(sub_rng.choice(len(base_train), args.subset_train, replace=False))
        else:
            active_train = np.arange(len(base_train))
        if args.subset_test is not None and args.subset_test < len(base_test):
            active_test = np.sort(sub_rng.choice(len(base_test), args.subset_test, replace=False))
        else:
            active_test = np.arange(len(base_test))

        n_per_batch = len(active_train) / args.k
        log.info("[seed %d] using %d train / %d test imgs (n_per_batch=%.1f, k=%d)",
                 seed, len(active_train), len(active_test), n_per_batch, args.k)

        for noise_level in noise_grid:
            sigma_meas_sq = float(noise_level ** 2)
            eta = 1.0 / sigma_meas_sq
            for sigma_M in sigma_M_grid:
                sigma_M = float(sigma_M)
                ratio = (sigma_M ** 2) / sigma_meas_sq
                cell_bar.set_postfix(seed=seed, noise=f"{noise_level:.3f}",
                                     sigma_M=f"{sigma_M:.3f}", ratio=f"{ratio:.2f}")
                log.info("=== cell %d/%d | seed=%d noise_level=%.4f sigma_M=%.4f ratio=%.3f ===",
                         len(rows) + 1, n_cells, seed, noise_level, sigma_M, ratio)
                mi = run_cell(base_train, base_test, active_train, active_test, img_shape,
                              seed, noise_level, sigma_M, num_classes, device, args)
                log.info("cell result: MI(pred;true) = %.4f bits", mi)
                rows.append(dict(
                    seed=seed,
                    noise_level=float(noise_level),
                    eta=float(eta),
                    sigma_meas_sq=sigma_meas_sq,
                    sigma_M=sigma_M,
                    sigma_M_sq_ratio=float(ratio),
                    sigma_M_sq=float(sigma_M ** 2),
                    n_per_batch=float(n_per_batch),
                    k=int(args.k),
                    MI=mi,
                    MI_std_if_estimator_provides_it=np.nan,
                ))
                # checkpoint after every cell so long runs are resumable-by-inspection
                pd.DataFrame(rows).to_csv(results_csv, index=False)
                cell_bar.update(1)
    cell_bar.close()

    # stash run metadata for the analysis step
    meta = dict(
        dataset=args.dataset,
        num_classes=num_classes,
        classifier='mobilenet_v3_small (IMAGENET1K_V1), final Linear -> num_classes',
        optimizer=f'Adam(lr={args.lr}), CrossEntropyLoss',
        epochs=args.epochs,
        size=args.size,
        img_shape=list(img_shape),
        k=args.k,
        seeds=list(args.seeds),
        noise_grid=list(map(float, noise_grid)),
        sigma_M_grid=list(map(float, sigma_M_grid)),
        sigma_meas_mid=sigma_meas_mid,
        ratio_targets=list(map(float, ratio_targets)),
        n_train=len(active_train),
        n_test=len(active_test),
        mi_estimator='latentmi.ksg.midd(Y_true, Y_pred) [bits]',
    )
    with open(os.path.join(args.output_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s and run_meta.json (%d rows)", results_csv, len(rows))

    # build figures + summary.json in the same job
    if not args.no_analyze:
        try:
            import analyze_batch_effect
            log.info("building figures + summary.json ...")
            summary = analyze_batch_effect.generate(args.output_dir)
            log.info("analysis done. summary:\n%s", json.dumps(summary, indent=2))
        except Exception as e:  # keep results.csv even if plotting fails
            log.exception("analysis step failed (results.csv is intact): %s", e)

    log.info("ALL DONE.")


if __name__ == '__main__':
    main()
