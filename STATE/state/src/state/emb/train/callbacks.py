import time
import logging
import math
from typing import Any, Optional

import torch
import lightning as L
from lightning.fabric.utilities.throughput import measure_flops

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StepBasedEarlyStopping(L.Callback):
    """Early-stop on a monitored metric using an optimizer-step window.

    Standard ``lightning.pytorch.callbacks.EarlyStopping`` fires on
    ``on_validation_end`` and counts patience in validation events, so its
    real-step horizon depends on ``val_check_interval`` AND on
    ``batches_per_epoch`` (Lightning clamps val frequency to the epoch
    when ``check_val_every_n_epoch`` is not None). On tiny datasets that
    silently means "stop after a handful of optimizer steps."

    This callback decouples from epochs and validation cadence: every
    ``every_n_steps`` optimizer steps it reads the latest cached value of
    ``monitor`` and applies patience in those step-windows. So with
    ``every_n_steps=500`` and ``patience=5``, training stops after 2,500
    consecutive optimizer steps with no improvement, regardless of N.

    The metric still has to come from somewhere — usually validation. As
    long as validation runs at least as often as the check window
    (e.g. ``val_check_interval=500`` for ``every_n_steps=500``) the cached
    value is fresh; on the first windows where the metric isn't available
    yet the check is a no-op.
    """

    def __init__(
        self,
        monitor: str = "validation/val_loss",
        every_n_steps: int = 500,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
        min_steps: int = 0,
    ):
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {every_n_steps}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        self.monitor = monitor
        self.every_n_steps = int(every_n_steps)
        self.patience = int(patience)
        self.min_delta = float(min_delta) if mode == "max" else -float(min_delta)
        self.mode = mode
        self.verbose = verbose
        # Warm-up window: ES checks are no-ops while global_step < min_steps,
        # so the model is guaranteed to train for at least min_steps optimizer
        # steps (typically one full epoch) before patience-based termination
        # can fire. Best-metric tracking starts fresh at step >= min_steps.
        self.min_steps = int(min_steps)
        self._best: float = math.inf if mode == "min" else -math.inf
        self._wait_count: int = 0
        self._stopped_step: int | None = None
        self._monitor_op = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def state_dict(self) -> dict:
        return {
            "best": self._best,
            "wait_count": self._wait_count,
            "stopped_step": self._stopped_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._best = state_dict.get("best", self._best)
        self._wait_count = state_dict.get("wait_count", 0)
        self._stopped_step = state_dict.get("stopped_step", None)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if step < self.min_steps:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        try:
            current = float(metric.item() if hasattr(metric, "item") else metric)
        except Exception:
            return
        if not math.isfinite(current):
            return

        improved = self._monitor_op(current - self.min_delta, self._best)
        if improved:
            if self.verbose:
                logger.info(
                    f"[StepBasedEarlyStopping] step={step} {self.monitor}={current:.6f} "
                    f"improved over best={self._best:.6f}; wait reset"
                )
            self._best = current
            self._wait_count = 0
        else:
            self._wait_count += 1
            if self.verbose:
                logger.info(
                    f"[StepBasedEarlyStopping] step={step} {self.monitor}={current:.6f} "
                    f"no improvement (best={self._best:.6f}); "
                    f"wait={self._wait_count}/{self.patience}"
                )
            if self._wait_count >= self.patience:
                self._stopped_step = step
                trainer.should_stop = True
                if self.verbose:
                    logger.info(
                        f"[StepBasedEarlyStopping] step={step} stopping: "
                        f"{self.patience} windows of {self.every_n_steps} steps "
                        f"with no improvement on {self.monitor}"
                    )


class LogLR(L.Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args,
    ) -> None:
        if trainer.global_rank == 0:
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"trainer/learning_rate": pl_module.lr_schedulers().get_last_lr()[0]},
                    step=trainer.global_step,
                )


class PerfProfilerCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time


class ProfilerCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []
        self.cfg = cfg

        self.profile_steps = cfg.experiment.profile.profile_steps

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        if batch_idx == self.profile_steps[0]:
            logging.info(f"Starting NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_push("VCIProfiledSection")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time

        if batch_idx == self.profile_steps[1]:
            logging.info(f"Stopping NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_pop()


class ResumeCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def on_train_start(self, trainer, pl_module):
        if self._cfg.optimizer.get("reset_lr_on_restart", False):
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    original_lr = param_group.get("lr", None)
                    param_group["lr"] = self._cfg.optimizer.max_lr
                    logging.info(f"Reset learning rate from {original_lr} to {param_group['lr']}")


class EMACallback(L.Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.beta = decay
        self.velocity = {}

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer):
        # Check if EMA is enabled via the config flag.
        if pl_module.cfg.model.get("ema", False):
            with torch.no_grad():
                for param in pl_module.parameters():
                    if param.grad is None:
                        continue

                    param_id = id(param)
                    if param_id not in self.velocity:
                        self.velocity[param_id] = torch.zeros_like(param.grad)

                    self.velocity[param_id] = self.beta * self.velocity[param_id] + (1 - self.beta) * param.grad
                    param.grad = self.velocity[param_id].clone()


class CumulativeFLOPSCallback(L.Callback):
    """
    PyTorch Lightning callback to track cumulative FLOPS during SE training.

    - Measures FLOPs once on the first training batch using `measure_flops`.
    - Tracks cumulative FLOPs and logs at validation frequency.
    - Logs cumulative_flops to trainer loggers (e.g., W&B, CSV) at validation cadence.

    Args:
        use_backward: If True, include backward pass FLOPs in the measurement.
    """

    def __init__(
        self,
        *,
        use_backward: bool = False,
    ) -> None:
        super().__init__()
        self.use_backward = use_backward

        self._flops_per_batch: Optional[int] = None
        self._measured: bool = False
        self._cumulative_flops: int = 0
        self._batch_count: int = 0

    def _trainstep_forward_backward(self, model: L.LightningModule, batch: Any) -> torch.Tensor:
        """Encapsulate calling StateEmbeddingModel.training_step and backward.

        This intentionally targets StateEmbeddingModel's signature and
        performs both forward and backward to capture full FLOPs.

        !!WARNING!!
        This has only been tested with StateEmbeddingModel. Behavior with any other model has not been verified.
        """
        model.zero_grad(set_to_none=True)
        loss: torch.Tensor = model.training_step(batch, 0)  # type: ignore
        if self.use_backward:
            loss.backward()
        return loss

    def _measure_flops_once(self, trainer: L.Trainer, pl_module: Any, batch: Any) -> None:
        if self._measured:
            return

        model = pl_module

        def forward_fn():
            return self._trainstep_forward_backward(model, batch)

        self._flops_per_batch = int(measure_flops(model, forward_fn=forward_fn))
        logger.info(f"CumulativeFLOPSCallback: Measured FLOPs per batch: {self._flops_per_batch}")

        model.zero_grad(set_to_none=True)
        self._measured = True

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: Any, batch: dict, batch_idx: int) -> None:
        if not self._measured and batch_idx == 0 and trainer.current_epoch == 0:
            self._measure_flops_once(trainer, pl_module, batch)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: dict, batch_idx: int) -> None:
        if self._flops_per_batch is None:
            return

        self._batch_count += 1
        self._cumulative_flops += self._flops_per_batch

        # Log cumulative FLOPs after every training batch
        pl_module.log(
            "cumulative_flops",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def on_validation_start(self, trainer: L.Trainer, pl_module: Any) -> None:
        if self._flops_per_batch is None:
            return

        # Log cumulative FLOPs at validation frequency for W&B panel alignment
        pl_module.log(
            "cumulative_flops_val_sync",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
