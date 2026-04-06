from scaling_laws.prepare.data import Experiments
import shutil
import os

src_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-08_22-10_download_data_to_retrain_scvi_models"
dst_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-08_22-15_reeval_scvis"

if not os.path.exists(dst_folder):
    shutil.copytree(src_folder, dst_folder)

qualities = [1.0, 0.5414548, 0.2931733, 0.1587401, 0.0859506, 0.0465384, 0.0251984, 0.0136438, 0.0073875, 0.004]

experiments: Experiments = Experiments(
    datasets=["shendure"],
    sizes=[10000000],
    qualities=qualities,
    algos=["SCVI"],
    path_to_data_dir=dst_folder,
    signal_columns=["author_day"],
    device=0,
    seed=42,
)

# SCVI loads models from the default location (base_dir/results/SCVI/model)
# Run all qualities in parallel
experiments.parallel_run(
    max_workers=30,  # SCVI needs to run alone since it blocks all GPUs
    sleep_time=0.2,
    retrain=False,
    reembed=True,
    recompute_mutual_information=True,
    mem_limit={"Geneformer": 23_000, "default": 23_000},
)

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)
