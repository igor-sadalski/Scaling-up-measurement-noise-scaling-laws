# Batch-effect pilot

Extends the tissuemnist image-noise MI experiment (`../run_kidney_experiment.py`) with a
simulated, **uncorrected batch effect** (an additive per-batch pixel shift) and tests
whether the resulting MI curves collapse onto the no-batch scaling curve under

$$\eta_{\mathrm{eff}} = \frac{\eta}{1 + \sigma_M^2/\sigma_{\mathrm{meas}}^2}.$$

## Mapping to the existing pipeline

The current experiment adds Gaussian measurement noise via `add_gauss(x, sigma)` =
`x + randn_like(x) * sigma`, inside the transform, at the loaded native resolution
(with `size=224` the `Resize` is an identity, so noise lives at 224×224). Therefore:

- `noise_level` == `sigma` == per-pixel measurement-noise **std**
- $\sigma_{\mathrm{meas}}^2 = \texttt{noise\_level}^2$
- We define the SNR knob $\eta = 1/\sigma_{\mathrm{meas}}^2$ (signal variance is a fixed
  dataset constant; it only rescales $\eta$ and cancels from the collapse test).

The batch shift $\mu_b \sim \mathcal N(0, \sigma_M^2 I)$ has the same shape as the
3-channel image and is added **after** the per-image measurement noise and **before** the
(identity) resize — same resolution and units as $\sigma_{\mathrm{meas}}$, so
$\sigma_M^2/\sigma_{\mathrm{meas}}^2$ is a well-defined per-pixel ratio. No batch
correction is applied and batch labels are never given to the classifier.

## Design

- **Batches:** $k=100$, each image assigned uniformly at random once per seed
  (train and test each split into $k$ batches, sharing the same $\mu_b$).
- **Sweep:** `noise_grid` (6 values of $\sigma_{\mathrm{meas}}$) × `ratio_targets`
  (6 values of $\sigma_M^2/\sigma_{\mathrm{meas}}^2$ **at the geometric middle** of the
  noise grid, including 0 → exact baseline). $\sigma_M=0$ rows *are* the baseline curve.
- **MI:** `latentmi.ksg.midd(Y_true, Y_pred)` (discrete MI between predicted and true
  labels, bits) — the existing label-MI estimator. It returns a point estimate, so
  `MI_std_if_estimator_provides_it` is NaN; spread comes from `--seeds`.

## Run

Uses the **`lt`** conda env (has torch + torchvision + medmnist; `latentmi` was pip-installed
into it with `--no-deps`).

**`batch_effect_experiment.py` is directly submittable** — it carries its own `#SBATCH`
header (partition `bates`, 1 GPU, 128 GB, 2 CPU, 12 h) and an env-pinned shebang
(`lt`'s python, so no `conda activate` needed), and it **auto-runs the analysis** (figures +
`summary.json`) at the end:

```bash
cd images/pilot
sbatch batch_effect_experiment.py                 # full default sweep, all-in-one
sbatch run_pilot.slurm                            # same, but 3 seeds / 30 epochs wrapper
```

Run it by hand too (logs stream with timestamps, tqdm bars over sweep cells / epochs / eval):

```bash
conda activate lt
cd images/pilot

# full pilot (slow: one MobileNet training per sweep cell)
python batch_effect_experiment.py --seeds 0 1 2 --epochs 30

# fast sanity run
python batch_effect_experiment.py --size 28 --epochs 2 \
    --subset-train 5000 --subset-test 5000 --seeds 0

# re-plot from an existing results.csv without re-running the sweep
python analyze_batch_effect.py --results-dir results
```

Pass `--no-analyze` to skip the figure step. Default `--num-workers` is 2 (matches the 2
requested CPUs).

Key flags: `--noise-grid`, `--ratio-targets`, `--k`, `--seeds`, `--epochs`,
`--subset-train/--subset-test` (subsample for speed), `--size`.

## Outputs (`results/`)

- `results.csv` — `seed, noise_level, eta, sigma_meas_sq, sigma_M, sigma_M_sq_ratio,
  sigma_M_sq, n_per_batch, k, MI, MI_std_if_estimator_provides_it` (checkpointed each cell).
- `fig1_raw_curves.png` — MI vs $\eta$ per $\sigma_M$; batch curves shift rightward.
- `fig2_collapse.png` — MI vs $\eta_{\mathrm{eff}}^{\mathrm{theory}}$; should overlay baseline.
- `fig3_empirical_map.png` — $\eta/\eta_{\mathrm{eff}}^{\mathrm{fit}}$ vs
  $\sigma_M^2/\sigma_{\mathrm{meas}}^2$ (fit by matching each batch MI to the baseline via
  monotone interpolation), with a line fit vs the theory line $1+x$.
- `run_meta.json`, `summary.json` — collapse-residual MSE per $\sigma_M$, fig3
  slope/intercept/$R^2$, and $n$, $k$, dataset, classifier details.

## Notes / deviations

- Measurement noise is deterministic per `(seed, image index)` so the whole sweep is
  reproducible; the original draws fresh noise each epoch (augmentation-like). This does
  not affect the variance relationship under test.
- $\mu_b = \sigma_M \cdot \text{base}_b$ with `base_b ~ N(0,I)` fixed per seed, so
  $\sigma_M=0$ is the exact baseline and larger $\sigma_M$ scales the same directions —
  isolating the magnitude effect the theory predicts.
