from scaling_laws.prepare.data import Experiments
import numpy as np

datasets = ["shendure"]
sizes = [10000000]
qualities = [0.0859506]
path_to_data_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19_48_download_necessary_models"
signal_columns = ["author_day"]
seeds = [42]

experiments: Experiments = Experiments(
    datasets=datasets,
    sizes=sizes,
    qualities=qualities,
    algos=["Geneformer", "PCA", "SCVI", "RandomProjection"],
    path_to_data_dir=path_to_data_dir,
    signal_columns=signal_columns,
    device=0,
    seed=42,
)

experiments.parallel_run(
    max_workers=100,
    sleep_time=0.2,
    retrain=False,
    reembed=True,
    recompute_mutual_information=True,
    mem_limit={"Geneformer": 23_000, "default": 23_000},
)
