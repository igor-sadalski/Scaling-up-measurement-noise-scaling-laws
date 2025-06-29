from scaling_laws.prepare.data import Experiments
import argparse
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sizes",
    type=int,
    nargs="+",
    default=[60_000],
    help="List of sizes to test",
)
parser.add_argument(
    "--qualities",
    type=float,
    nargs="+",
    default=[1.0],
    help="List of quality values to test",
)
parser.add_argument(
    "--algos",
    type=str,
    nargs="+",
    default=["PCA"],
    help="List of algorithms to test",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="/home/jupyter/igor_repos/noise_scaling_laws/data/",
    help="Base directory to save the data",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Device to use",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=3,
    help="Max epochs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
parser.add_argument(
    "--early_stopping_patience",
    type=int,
    # required=True,
    default=0,
    help="Early stopping patience",
)
parser.add_argument(
    "--dataset",
    type=str,
    # required=True,
    default="merfish",
    help="Dataset to use",
)
parser.add_argument(
    "--signal_columns",
    type=str,
    nargs="*",
    default=["author_day", "author_cell_type", "donor_id"],
    help="Signal columns to use",
)
parser.add_argument(
    "--retrain",
    type=str2bool,
    default=True,
    help="Whether to retrain the model",
)
parser.add_argument(
    "--reembed",
    type=str2bool,
    default=True,
    help="Whether to re-embed the data",
)
parser.add_argument(
    "--recompute_mutual_information",
    type=str2bool,
    default=True,
    help="Whether to recompute mutual information",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to checkpoint directory to resume training from",
)
parser.add_argument(
    "--reembed_checkpoint",
    type=str,
    default="model",
    help="Path to checkpoint to use for reembedding",
)
parser.add_argument(
    "--batch_size_inference",
    type=int,
    default=None,
    help="Batch size for inference",
)
args = parser.parse_args()
experiments = Experiments(
    datasets=[args.dataset],
    sizes=args.sizes,
    qualities=args.qualities,
    algos=args.algos,
    path_to_data_dir=args.base_dir,
    signal_columns=args.signal_columns,
    device=args.device,
    seed=args.seed,
)
experiments.single_job(
    dataset=args.dataset,
    size=args.sizes[0],
    quality=args.qualities[0],
    algo=args.algos[0],
    max_epochs=args.max_epochs,
    early_stopping_patience=args.early_stopping_patience,
    device=args.device,
    retrain=args.retrain,
    reembed=args.reembed,
    recompute_mutual_information=args.recompute_mutual_information,
    checkpoint_path=args.checkpoint_path,
    reembed_checkpoint=args.reembed_checkpoint,
    batch_size_inference=args.batch_size_inference,
)
