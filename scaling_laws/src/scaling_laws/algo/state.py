import os
import re
import glob
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .abc import BaseAlgorithm


class State(BaseAlgorithm):
    """STATE SE (State Embedding) algorithm.

    Self-supervised cell embedding model using a transformer encoder.
    Architecture matches Geneformer: 256 hidden, 4 heads, 3 layers, 512 context.
    Uses ESM-2 protein embeddings by default (via merged_esm_embeddings.pt);
    falls back to one-hot gene embeddings if no ESM file is found.

    All STATE operations run via the `state` CLI in the `state` conda env:
      - state emb preprocess  (build gene-embedding profile)
      - state emb fit          (train the model)
      - state emb transform    (compute embeddings)
    """

    def __init__(
        self,
        base_dir: str,
        device: int = 0,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        max_steps: int | None = None,
        dataset_name: str | None = None,
        seed: int = 42,
        pad_length: int = 2048,
        emsize: int = 256,
        d_hid: int = 512,
        nhead: int = 4,
        nlayers: int = 3,
        output_dim: int = 256,
        batch_size: int = 64,
        max_lr: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__(base_dir, device, model_name="model", seed=seed)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.max_steps = max_steps
        self.dataset_name = dataset_name or "unknown"
        self.pad_length = pad_length
        self.emsize = emsize
        self.d_hid = d_hid
        self.nhead = nhead
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.dropout = dropout

        # STATE env paths — sourced from scaling_laws.paths so they're
        # overridable via env vars (see paths.py / README env-var contract).
        from scaling_laws.paths import STATE_PYTHON, STATE_PACKAGE_DIR, STATE_DEFAULTS_YAML
        self.state_python = STATE_PYTHON
        self.state_package_dir = STATE_PACKAGE_DIR
        self.state_defaults_yaml = STATE_DEFAULTS_YAML

        # Preprocessing profile goes into preprocessed/state_data/ for each split
        self.profile_dir = self.train_data_path / "state_data"
        self.val_state_data_dir = self.validation_data_path / "state_data"
        self.test_state_data_dir = self.test_data_path / "state_data"

        # Config lives alongside the profile in state_data/; checkpoints under results/
        self.config_path = self.profile_dir / "state_config.yaml"
        self.checkpoint_dir = self.save_folder_path / self.model_name / "checkpoints"
        self.profile_name = f"scaling_{self.dataset_name}_{self.size}_{self.quality}".replace(".", "_")

    def _get_env(self) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        return env

    def _get_unique_port(self) -> int:
        """Find a free port to avoid EADDRINUSE conflicts between concurrent jobs."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _check_preprocessed(self) -> None:
        """Verify that the preprocessing profile exists.

        The profile must be created beforehand via
        ``Experiments.prepare_state_data()``.  If it is missing this
        raises ``FileNotFoundError`` instead of silently re-running
        preprocessing.
        """
        profile_marker = self.profile_dir / f"all_embeddings_{self.profile_name}.pt"
        if not profile_marker.exists():
            raise FileNotFoundError(
                f"State preprocessing profile not found at {profile_marker}. "
                f"Run Experiments.prepare_state_data() first."
            )

    def train(self) -> None:
        import math
        import yaml

        self._check_preprocessed()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # If a `last.ckpt` from a previous (interrupted) run exists under the
        # profile-specific run dir, preserve the directory so Lightning can
        # auto-resume via ``get_latest_checkpoint`` (STATE/emb/utils.py). In
        # that case we also skip the wipe of sibling version_* / metrics.csv
        # artifacts so the resumed session can stitch its own new CSVLogger
        # version alongside them. Otherwise, clear stale state.
        run_dir = self.checkpoint_dir / f"state_{self.profile_name}"
        resume_ckpt = run_dir / "last.ckpt"
        if resume_ckpt.exists():
            print(f"  Resuming from {resume_ckpt} (checkpoint dir preserved)")
        else:
            for item in self.checkpoint_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        env = self._get_env()
        train_h5ad = self.train_data_path / "preprocessed.h5ad"

        adata = ad.read_h5ad(train_h5ad, backed="r")
        num_cells = int(adata.shape[0])
        batches_per_epoch = max(1, num_cells // self.batch_size)
        # Validate every 1000 optimizer steps regardless of dataset size.
        # STATE's trainer sets ``check_val_every_n_epoch=None`` so this is a
        # pure step-window — tiny datasets stretch validation across many
        # epochs (e.g. N=100 -> 1 batch/epoch -> val every 1000 epochs).
        # ``StepBasedEarlyStopping`` (in STATE) consumes the same window.
        val_interval = 1000

        # Determine training duration.  Early stopping (patience=5) is ALWAYS
        # enabled — `max_steps` / `max_epochs` act as an upper bound, not a
        # replacement.
        if self.max_steps is not None:
            max_epochs = max(1, math.ceil(self.max_steps / batches_per_epoch))
            print(f"  Data: {num_cells} cells, ~{batches_per_epoch} batches/epoch, val every {val_interval} steps")
            print(f"  Step budget: {self.max_steps} steps → {max_epochs} epochs (with early stopping after {batches_per_epoch}-step warm-up = ≥1 epoch)")
        else:
            max_epochs = self.max_epochs
            print(f"  Data: {num_cells} cells, ~{batches_per_epoch} batches/epoch, val every {val_interval} steps")
            print(f"  Epoch budget: {max_epochs} epochs (with early stopping after {batches_per_epoch}-step warm-up = ≥1 epoch)")

        # state emb fit — train with Hydra overrides
        cmd_fit = [
            str(self.state_python), "-m", "state", "emb", "fit",
            "--conf", str(self.config_path),
            # Profile selection
            f"embeddings.current={self.profile_name}",
            f"dataset.current={self.profile_name}",
            # Dataset
            f"dataset.num_cells={num_cells}",
            "dataset.num_train_workers=4",
            "dataset.num_val_workers=2",
            f"dataset.pad_length={self.pad_length}",
            f"dataset.P={self.pad_length // 4}",
            f"dataset.N={self.pad_length // 4}",
            f"dataset.S={self.pad_length // 4}",
            # Model architecture
            f"model.batch_size={self.batch_size}",
            f"model.emsize={self.emsize}",
            f"model.d_hid={self.d_hid}",
            f"model.nhead={self.nhead}",
            f"model.nlayers={self.nlayers}",
            f"model.output_dim={self.output_dim}",
            "model.dataset_correction=false",
            f"model.dropout={self.dropout}",
            # Optimizer
            f"optimizer.max_lr={self.max_lr}",
            "optimizer.gradient_accumulation_steps=1",
            "optimizer.weight_decay=0.01",
            # Experiment
            f"experiment.name=state_{self.profile_name}",
            f"experiment.num_epochs={max_epochs}",
            "experiment.num_gpus_per_node=1",
            "experiment.num_nodes=1",
            f"experiment.port={self._get_unique_port()}",
            f"experiment.val_check_interval={val_interval}",
            # Log train loss every 10 optimizer steps — cuts metrics.csv rows
            # by ~10x without losing the shape of the training curve.  Val is
            # already at ~1k steps via `val_check_interval` above.
            "+experiment.log_every_n_steps=10",
            "experiment.limit_val_batches=50",
            # Checkpoint: save a .ckpt every 10k optimizer steps (independent
            # of the 1k-step validation cadence) and always maintain a
            # ``last.ckpt`` pointer so training can resume after a crash. Keep
            # all 10k-step ckpts (``save_top_k=-1``); the best is still
            # selected downstream by parsing metrics.csv for argmin(val_loss)
            # and matching the step to a .ckpt filename (see
            # ``_find_best_checkpoint`` below). Per-run footprint drops ~10×
            # vs. the prior 1k-step cadence — needed to fit the 10M-cell sweep
            # on a 6.8T nvme without filling the disk.
            f"experiment.checkpoint.path={self.checkpoint_dir}",
            "experiment.checkpoint.every_n_train_steps=10000",
            "experiment.checkpoint.monitor=validation/val_loss",
            "experiment.checkpoint.save_top_k=-1",
            "+experiment.checkpoint.save_last=true",
            # Disable logging and extra validations
            "wandb.enable=false",
            "validations.diff_exp.enable=false",
            "validations.perturbation.enable=false",
        ]

        # Early stopping is ALWAYS on for STATE runs (patience=5 unless the
        # caller overrides ``early_stopping_patience``).  A patience of 0 would
        # mean "stop the first time val_loss doesn't strictly improve", which
        # is never what we want, so clamp to a sensible minimum.
        es_patience = max(self.early_stopping_patience or 0, 5)
        # Early-stopping warm-up: don't allow ES to fire before one full epoch
        # (≈ batches_per_epoch optimizer steps) of training has elapsed. This
        # prevents large datasets from terminating mid-first-epoch when the
        # 5-window patience burns through faster than the data can be seen
        # once. After the warm-up, ES resumes its normal step-window behavior.
        min_es_steps = batches_per_epoch

        # ``every_n_steps`` and ``min_steps`` are NOT keys in state-defaults.yaml,
        # so passing them as CLI overrides crashes Hydra's compose step (in
        # ``state/__main__.py``); the ``+`` prefix that Hydra needs to add new
        # keys is in turn mis-parsed by ``OmegaConf.from_dotlist`` in
        # ``_cli/_emb/_fit.py`` (treats ``+`` as a literal char, putting the
        # value in a phantom ``+experiment`` branch). Workaround: inject these
        # two leaves into the per-job state_config.yaml that ``_fit.py`` loads
        # via ``OmegaConf.load``, bypassing both parsers.
        with open(self.config_path, "r") as f:
            cfg_yaml = yaml.safe_load(f)
        cfg_yaml.setdefault("experiment", {}).setdefault("early_stopping", {})
        cfg_yaml["experiment"]["early_stopping"]["every_n_steps"] = val_interval
        cfg_yaml["experiment"]["early_stopping"]["min_steps"] = min_es_steps
        with open(self.config_path, "w") as f:
            yaml.safe_dump(cfg_yaml, f, sort_keys=False)

        cmd_fit.extend([
            "experiment.early_stopping.enable=true",
            "experiment.early_stopping.monitor=validation/val_loss",
            f"experiment.early_stopping.patience={es_patience}",
            "experiment.early_stopping.min_delta=0.0",
            "experiment.early_stopping.mode=min",
        ])
        print(f"  Running: {' '.join(cmd_fit)}")
        try:
            subprocess.run(cmd_fit, cwd=str(self.state_package_dir), env=env, check=True)
            print("  Training complete.")
        finally:
            # Always persist loss curves and the final train/val/test loss
            # scalars — even on crash / early SIGTERM — so nothing is lost.
            self._save_loss_curves()
            self._save_final_losses()

    def _find_lightning_metrics_csv(self) -> Path | None:
        """Locate the CSVLogger metrics.csv produced by STATE's Lightning trainer.

        Lightning's RobustCSVLogger writes to
        ``{checkpoint_dir}/state_{profile_name}/version_N/metrics.csv``.
        We pick the most recent non-empty version dir and prefer the one whose
        metrics.csv actually has both train and val columns.
        """
        run_dir = self.checkpoint_dir / f"state_{self.profile_name}"
        if not run_dir.is_dir():
            return None
        version_dirs = sorted(
            (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("version_")),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        )
        best: Path | None = None
        for vd in version_dirs:
            m = vd / "metrics.csv"
            if not m.exists() or m.stat().st_size == 0:
                continue
            try:
                cols = pd.read_csv(m, nrows=0).columns
            except Exception:
                continue
            has_step = "step" in cols
            has_train = "trainer/train_loss" in cols
            has_val = "validation/val_loss" in cols
            if has_step and has_train and has_val:
                best = m  # keep scanning — later versions win
            elif best is None and has_step:
                best = m
        return best

    def _save_loss_curves(self) -> None:
        """Copy Lightning metrics.csv to ``results/State/loss/metrics.csv``.

        The CSV contains per-optimizer-step rows with columns ``step``,
        ``epoch``, ``trainer/train_loss`` and ``validation/val_loss`` (the
        latter populated only on val_check_interval steps). Safe to call
        multiple times; silently no-ops if no metrics file is produced.
        """
        import shutil

        metrics_file = self._find_lightning_metrics_csv()
        if metrics_file is None:
            print(f"  Warning: no Lightning metrics.csv under {self.checkpoint_dir}/state_{self.profile_name}/version_*; "
                  f"train/val loss curves will NOT be saved.")
            return

        loss_dir = self.save_folder_path / self.model_name / "loss"
        loss_dir.mkdir(parents=True, exist_ok=True)
        dest = loss_dir / "metrics.csv"
        shutil.copy(metrics_file, dest)

        try:
            df = pd.read_csv(dest)
            n_train = df["trainer/train_loss"].notna().sum() if "trainer/train_loss" in df.columns else 0
            n_val = df["validation/val_loss"].notna().sum() if "validation/val_loss" in df.columns else 0
            last_step = int(df["step"].max()) if "step" in df.columns and not df.empty else 0
            print(f"  Loss curves saved to {dest} "
                  f"(train_pts={n_train}, val_pts={n_val}, last_step={last_step})")
        except Exception as e:
            print(f"  Loss curves saved to {dest} (could not summarize: {e})")

    def _save_final_losses(self) -> None:
        """Persist final train / val / test loss scalars for this run.

        Reads ``results/State/model/loss/metrics.csv`` (written by
        ``_save_loss_curves``) and writes three one-number text files into the
        model directory:

        - ``train_loss.txt`` — minimum of ``trainer/train_loss``
        - ``val_loss.txt``   — minimum of ``validation/val_loss``
        - ``test_loss.txt``  — same as ``val_loss.txt`` (proxy, matching the
          convention used in ``analysis/2026-04-15_14-43_compute_loss_scaling.py``)

        Safe to call on incomplete runs: whichever column is missing is skipped.
        """
        metrics_path = self.save_folder_path / self.model_name / "loss" / "metrics.csv"
        if not metrics_path.exists():
            print(f"  Warning: {metrics_path} missing — final losses NOT saved.")
            return
        try:
            df = pd.read_csv(metrics_path, on_bad_lines="skip")
        except Exception as e:
            print(f"  Warning: could not read {metrics_path}: {e}")
            return

        model_dir = self.save_folder_path / self.model_name
        train_loss_path = model_dir / "train_loss.txt"
        val_loss_path = model_dir / "val_loss.txt"

        def _best(col: str) -> float | None:
            if col not in df.columns:
                return None
            vals = df[col].dropna()
            return float(vals.min()) if not vals.empty else None

        best_train = _best("trainer/train_loss")
        best_val = _best("validation/val_loss")

        if best_train is not None:
            train_loss_path.write_text(f"{best_train:.6f}")
            print(f"  Final train loss: {best_train:.6f} → {train_loss_path}")
        if best_val is not None:
            val_loss_path.write_text(f"{best_val:.6f}")
            # test_loss.txt mirrors val_loss — matches the proxy convention in
            # analysis/2026-04-15_14-43_compute_loss_scaling.py.
            self.test_loss_path.write_text(f"{best_val:.6f}")
            print(f"  Final val loss  : {best_val:.6f} → {val_loss_path}")
            print(f"  Final test loss : {best_val:.6f} → {self.test_loss_path}  (proxy = best val_loss)")

    def compute_test_loss(self) -> float:
        """Return best val loss (proxy for test), writing all three scalar files.

        Delegates to ``_save_final_losses`` and then reads back test_loss.txt so
        callers that import ``compute_test_loss`` get a single scalar while the
        train / val / test files on disk stay in sync.
        """
        self._save_final_losses()
        if not self.test_loss_path.exists():
            raise FileNotFoundError(
                f"STATE {self.test_loss_path} not found — training probably produced no val_loss."
            )
        return float(self.test_loss_path.read_text().strip())

    def _find_best_checkpoint(self) -> str:
        """Select the best checkpoint when every validation step is saved.

        With ``save_top_k=-1`` Lightning keeps one .ckpt per validation step
        (filename contains ``step={N}``). We find argmin(val_loss) in the
        Lightning metrics CSV and match its step to the corresponding .ckpt.
        If the metrics CSV is unavailable we fall back to the latest .ckpt,
        then to the end-of-training ``_final.pt`` dump.
        """
        ckpts = sorted(glob.glob(str(self.checkpoint_dir / "**" / "*.ckpt"), recursive=True))
        ckpts = [c for c in ckpts if not c.endswith("last.ckpt")]

        def _step_of(path: str) -> int | None:
            m = re.search(r"step=(\d+)", os.path.basename(path))
            return int(m.group(1)) if m else None

        # Preferred: pick the ckpt whose step matches argmin(val_loss).
        if ckpts:
            metrics_csv = self._find_lightning_metrics_csv()
            if metrics_csv is not None:
                try:
                    df = pd.read_csv(metrics_csv, on_bad_lines="skip")
                    if {"step", "validation/val_loss"}.issubset(df.columns):
                        val = df[["step", "validation/val_loss"]].dropna()
                        if not val.empty:
                            best_step = int(val.loc[val["validation/val_loss"].idxmin(), "step"])
                            by_step = {_step_of(c): c for c in ckpts if _step_of(c) is not None}
                            if best_step in by_step:
                                return by_step[best_step]
                            # Nearest-earlier step fallback (val fires off a
                            # slightly different boundary than checkpoint saves
                            # in some Lightning versions).
                            earlier = [s for s in by_step if s is not None and s <= best_step]
                            if earlier:
                                return by_step[max(earlier)]
                except Exception as exc:
                    print(f"  Warning: could not pick best checkpoint via metrics.csv ({exc}); "
                          "falling back to latest .ckpt.")

            # Fallback: latest step on disk.
            stepped = [(s, c) for c in ckpts if (s := _step_of(c)) is not None]
            if stepped:
                return max(stepped, key=lambda t: t[0])[1]
            return ckpts[-1]

        # No .ckpt files — use the end-of-training _final.pt the trainer writes.
        finals = sorted(glob.glob(str(self.checkpoint_dir / "*_final.pt")))
        if finals:
            return finals[-1]

        raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

    def embed(self) -> np.ndarray:
        ckpt_path = self._find_best_checkpoint()
        print(f"  Using checkpoint: {ckpt_path}")

        test_h5ad = str(self.test_data_path / "preprocessed.h5ad")
        npy_output = str(self.save_folder_path / self.model_name / "embeddings.npy")

        # state emb transform — compute embeddings from checkpoint
        cmd = [
            str(self.state_python), "-m", "state", "emb", "transform",
            "--checkpoint", ckpt_path,
            "--input", test_h5ad,
            "--output", npy_output,
        ]
        env = self._get_env()
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(self.state_package_dir), env=env, check=True)

        embeddings = np.load(npy_output)
        pd.DataFrame(embeddings).to_csv(self.embeddings_path, index=False)
        print(f"  Embeddings: {embeddings.shape}")
        return embeddings
