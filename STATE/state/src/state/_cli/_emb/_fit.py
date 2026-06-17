import argparse as ap


def add_arguments_fit(parser: ap.ArgumentParser):
    """Add arguments for embedding training CLI."""
    parser.add_argument("--conf", type=str, default=None, help="Path to config YAML file")
    parser.add_argument(
        "hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., embeddings.current=esm2-cellxgene)"
    )


def run_emb_fit(cfg, args):
    """
    Run state training with the provided config and overrides.
    """
    import logging
    import os
    import sys

    # Lightning 2.6+ calls torch.load(ckpt_path, weights_only=None) on resume,
    # and torch 2.6+ treats None-or-True as "safe mode" which rejects the
    # pickled ModelCheckpoint metadata (class refs) inside Lightning ckpts.
    # `setdefault` isn't enough because weights_only=None is explicitly passed
    # — we have to overwrite it. These are our own ckpts, so forcing False in
    # the STATE subprocess is safe.
    import torch as _torch
    _orig_torch_load = _torch.load
    def _loose_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_torch_load(*a, **kw)
    _torch.load = _loose_load

    from omegaconf import OmegaConf

    from ...emb.train.trainer import main as trainer_main

    log = logging.getLogger(__name__)

    # Load the base configuration
    if args.conf:
        cfg = OmegaConf.load(args.conf)

    # Process the remaining command line arguments as overrides
    if args.hydra_overrides:
        overrides = OmegaConf.from_dotlist(args.hydra_overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    # Validate required configuration
    if cfg.embeddings.current is None:
        log.error("Gene embeddings are required for training. Please set 'embeddings.current'")
        sys.exit(1)

    if cfg.dataset.current is None:
        log.error("Please set the desired dataset to 'dataset.current'")
        sys.exit(1)

    # Set environment variables
    os.environ["MASTER_PORT"] = str(cfg.experiment.port)
    # WAR: Workaround for sbatch failing when --ntasks-per-node is set.
    # lightning expects this to be set.
    os.environ["SLURM_NTASKS_PER_NODE"] = str(cfg.experiment.num_gpus_per_node)

    log.info(f"*************** Training {cfg.experiment.name} ***************")
    log.info(OmegaConf.to_yaml(cfg))

    # Execute the main training logic
    trainer_main(cfg)
