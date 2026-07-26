#!/mnt/home/gokulg/miniconda3/envs/lt/bin/python
#SBATCH --job-name=bn_ablation
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot/slurm_logs/%x_%j.err
#
# bn_ablation_experiment.py
#
# Diagnostic for the batch-iid equivalence failure: at a few failing cells, compare MI for
#   shift    -- condition A: measurement noise + coherent per-batch shift (current pipeline, BN)
#   shift_gn -- condition A but BatchNorm -> GroupNorm (kills batch-level normalization)
#   iid      -- condition B: pure iid noise of matched total variance sigma_meas^2+sigma_M^2
# If shift >> iid (expected) and shift_gn drops toward iid, batch-level BN is the remover.
# A handful of runs; NOT the full grid.

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import medmnist
from medmnist import INFO

PILOT_DIR = Path('/mnt/home/gokulg/scaling/Scaling-up-measurement-noise-scaling-laws/images/pilot')
if not PILOT_DIR.exists():
    PILOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PILOT_DIR.parent
for _d in (str(PILOT_DIR), str(IMAGES_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from run_kidney_experiment import create_model, train_model  # noqa: E402
from batch_effect_experiment import (  # noqa: E402
    BatchEffectDataset, assign_batches, make_batch_shifts,
    recalibrate_bn, evaluate_label_mi)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
log = logging.getLogger('bn_ablation')


def pick_groups(c):
    for g in (32, 16, 8, 4, 2):
        if c % g == 0:
            return g
    return 1


def bn_to_gn(module):
    """Replace every BatchNorm2d with a GroupNorm carrying the same affine params."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            gn = nn.GroupNorm(pick_groups(c), c, eps=child.eps, affine=True)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            bn_to_gn(child)


def run_one(mode, nl, sM, seed, base_train, base_test, active_train, active_test,
            img_shape, num_classes, device, args):
    # iid: fold the shift variance into the measurement noise, no shift
    if mode == 'iid':
        noise_used, shift_used = float(np.sqrt(nl ** 2 + sM ** 2)), 0.0
    else:
        noise_used, shift_used = float(nl), float(sM)

    split_rng = np.random.default_rng(seed)
    bt = assign_batches(split_rng, len(active_train), args.k)
    bv = assign_batches(split_rng, len(active_test), args.k)
    shifts = make_batch_shifts(seed, args.k, img_shape, shift_used)

    train_ds = BatchEffectDataset(base_train, active_train, bt, shifts, noise_used, args.size, noise_seed=seed)
    test_ds = BatchEffectDataset(base_test, active_test, bv, shifts, noise_used, args.size, noise_seed=seed + 1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    torch.manual_seed(seed * 100003 + hash((round(noise_used, 6), round(shift_used, 6), mode)) % 100003)
    model = create_model(num_classes, device)
    if mode == 'shift_gn':
        bn_to_gn(model)
        model = model.to(device)

    log.info("mode=%s seed=%d cell(sigma_meas=%.3f,sigma_M=%.3f) -> noise=%.4f shift=%.4f, %d epochs",
             mode, seed, nl, sM, noise_used, shift_used, args.epochs)
    t0 = time.time()
    train_model(model, train_dl, None, device, n_epochs=args.epochs, lr=args.lr)
    recalibrate_bn(model, train_dl, device, max_batches=args.bn_recalib_batches)  # no-op without BN
    mi, acc = evaluate_label_mi(model, test_dl, device)
    log.info("  -> MI=%.4f acc=%.3f (%.1fs)", mi, acc, time.time() - t0)
    return dict(mode=mode, seed=seed, sigma_meas=float(nl), sigma_M=float(sM),
                ratio=float(sM ** 2 / nl ** 2), noise_level_used=noise_used,
                sigma_M_used=shift_used, eta_eff=float(1.0 / (nl ** 2 + sM ** 2)),
                MI=float(mi), test_acc=float(acc))


def parse_cells(specs):
    out = []
    for s in specs:
        nl, sM = s.split(',')
        out.append((float(nl), float(sM)))
    return out


def main():
    p = argparse.ArgumentParser(description="BatchNorm ablation for the batch-effect diagnostic.")
    p.add_argument('--dataset', default='tissuemnist')
    p.add_argument('--output-dir', default=str(PILOT_DIR / 'results'))
    p.add_argument('--cells', nargs='+', default=['0.05,1.131', '0.1,1.131'],
                   help='failing cells as "sigma_meas,sigma_M"')
    p.add_argument('--modes', nargs='+', default=['shift', 'shift_gn', 'iid'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--bn-recalib-batches', type=int, default=100)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--subset-train', type=int, default=10000)
    p.add_argument('--subset-test', type=int, default=10000)
    p.add_argument('--no-download', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cells = parse_cells(args.cells)
    log.info("device=%s | cells=%s | modes=%s | seeds=%s", device, cells, args.modes, args.seeds)

    info = INFO[args.dataset]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    base_train = DataClass(split='train', transform=None, download=not args.no_download, size=args.size)
    base_test = DataClass(split='test', transform=None, download=not args.no_download, size=args.size)

    x0 = transforms.ToTensor()(base_train[0][0])
    if x0.shape[0] == 1:
        x0 = x0.repeat(3, 1, 1)
    img_shape = tuple(x0.shape)

    # subset fixed per seed (reuse the sweep's convention)
    subsets = {}
    for seed in args.seeds:
        r = np.random.default_rng(seed + 12345)
        at = (np.sort(r.choice(len(base_train), args.subset_train, replace=False))
              if args.subset_train < len(base_train) else np.arange(len(base_train)))
        te = (np.sort(r.choice(len(base_test), args.subset_test, replace=False))
              if args.subset_test < len(base_test) else np.arange(len(base_test)))
        subsets[seed] = (at, te)

    rows = []
    out_csv = os.path.join(args.output_dir, 'bn_ablation.csv')
    for (nl, sM) in cells:
        for mode in args.modes:
            for seed in args.seeds:
                at, te = subsets[seed]
                rows.append(run_one(mode, nl, sM, seed, base_train, base_test, at, te,
                                    img_shape, num_classes, device, args))
                pd.DataFrame(rows).to_csv(out_csv, index=False)  # checkpoint

    # summary: mean MI/acc per (cell, mode)
    df = pd.DataFrame(rows)
    summ = (df.groupby(['sigma_meas', 'sigma_M', 'mode'], as_index=False)
              .agg(MI=('MI', 'mean'), acc=('test_acc', 'mean'), n=('MI', 'size')))
    log.info("wrote %s (%d rows)\n%s", out_csv, len(rows), summ.to_string(index=False))
    with open(os.path.join(args.output_dir, 'bn_ablation_summary.json'), 'w') as f:
        json.dump(summ.to_dict(orient='records'), f, indent=2)
    log.info("interpretation: shift >> iid confirms A!=B; if shift_gn drops toward iid, "
             "batch-level BatchNorm is removing the coherent shift.")


if __name__ == '__main__':
    main()
