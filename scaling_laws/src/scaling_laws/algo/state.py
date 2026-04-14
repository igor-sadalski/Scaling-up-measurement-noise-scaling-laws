import os
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
        dataset_name: str | None = None,
        seed: int = 42,
    ):
        super().__init__(base_dir, device, model_name="model", seed=seed)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.dataset_name = dataset_name or "unknown"

        # STATE env paths
        self.state_python = Path("/home/igor/miniconda3/envs/state/bin/python")
        self.state_package_dir = Path("/home/igor/noise_scaling/modeling/STATE/state")
        self.state_defaults_yaml = self.state_package_dir / "src" / "state" / "configs" / "state-defaults.yaml"

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
        self._check_preprocessed()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Clear old checkpoints
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        env = self._get_env()
        train_h5ad = self.train_data_path / "preprocessed.h5ad"

        adata = ad.read_h5ad(train_h5ad, backed="r")
        num_cells = int(adata.shape[0])
        batches_per_epoch = max(1, num_cells // 128)
        # Match Geneformer: validate & checkpoint every 1000 steps,
        # capped at batches_per_epoch (Lightning requires val_check_interval <= training batches)
        val_interval = min(1000, batches_per_epoch)
        print(f"  Data: {num_cells} cells, ~{batches_per_epoch} batches/epoch, val every {val_interval} steps")

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
            "dataset.pad_length=512",
            "dataset.P=128",
            "dataset.N=128",
            "dataset.S=128",
            # Model architecture (Geneformer-scale: 256 hidden, 4 heads, 3 layers; learning dynamics from STATE defaults)
            "model.batch_size=128",
            "model.emsize=256",
            "model.d_hid=512",
            "model.nhead=4",
            "model.nlayers=3",
            "model.output_dim=256",
            "model.dataset_correction=false",
            "model.dropout=0.1",
            # Optimizer
            "optimizer.max_lr=5.0e-04",
            "optimizer.gradient_accumulation_steps=1",
            "optimizer.weight_decay=0.01",
            # Experiment
            f"experiment.name=state_{self.profile_name}",
            f"experiment.num_epochs={self.max_epochs}",
            "experiment.num_gpus_per_node=1",
            "experiment.num_nodes=1",
            f"experiment.port={self._get_unique_port()}",
            f"experiment.val_check_interval={val_interval}",
            "experiment.limit_val_batches=50",
            # Checkpoint: keep only the single best model by val_loss
            f"experiment.checkpoint.path={self.checkpoint_dir}",
            f"experiment.checkpoint.every_n_train_steps={val_interval}",
            "experiment.checkpoint.monitor=validation/val_loss",
            "experiment.checkpoint.save_top_k=1",
            "+experiment.checkpoint.save_last=false",
            # Early stopping
            "experiment.early_stopping.enable=true",
            "experiment.early_stopping.monitor=validation/val_loss",
            f"experiment.early_stopping.patience={self.early_stopping_patience}",
            "experiment.early_stopping.min_delta=0.0",
            "experiment.early_stopping.mode=min",
            # Disable logging and extra validations
            "wandb.enable=false",
            "validations.diff_exp.enable=false",
            "validations.perturbation.enable=false",
        ]
        print(f"  Running: {' '.join(cmd_fit)}")
        subprocess.run(cmd_fit, cwd=str(self.state_package_dir), env=env, check=True)
        print("  Training complete.")

    def _find_best_checkpoint(self) -> str:
        """Find the best checkpoint (single .ckpt saved by save_top_k=1)."""
        ckpts = sorted(glob.glob(str(self.checkpoint_dir / "**" / "*.ckpt"), recursive=True))
        # Prefer the best checkpoint saved by ModelCheckpoint, skip last.ckpt
        best_ckpts = [c for c in ckpts if not c.endswith("last.ckpt")]
        if best_ckpts:
            return best_ckpts[-1]
        if ckpts:
            return ckpts[-1]

        # Fallback: _final.pt written by the trainer at the end of training
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
