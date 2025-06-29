from scaling_laws.prepare.data import Experiments
import numpy as np

datasets = ["shendure"]
sizes = list(map(int, np.logspace(2, 7, 10)))
qualities = list(map(lambda x: round(x, 7), np.logspace(0, np.log10(10 / 2500), 10)))
path_to_data_dir = "/home/jupyter/igor_repos/noise_scaling_laws/data/"
signal_columns = ["author_day"]
seeds = [42]

# experiments: Experiments = Experiments(
#     datasets=datasets,
#     sizes=sizes,
#     qualities=qualities,
#     algos=["Geneformer"],
#     path_to_data_dir=path_to_data_dir,
#     signal_columns=signal_columns,
#     seed=seeds[0],
# )

# experiments.make_training_set(
#     remove_old=False,
#     download_raw_data=True,
#     sample_dataset=True,
#     downsample_dataset=True,
# )
# experiments.make_validation_set(remove_old=False)

# experiments.make_test_set(remove_old=False)

# experiments.tokenize_median_files(
#     partition_size=100_000,
#     chunk_size=100_000,
#     nproc=1,
#     recompute_median_files=True,
#     train=True,
#     validation=True,
#     test=True,
# )

for seed in seeds:

    experiments: Experiments = Experiments(
        datasets=datasets,
        sizes=sizes,
        qualities=qualities,
        algos=["RandomProjection"],
        path_to_data_dir=path_to_data_dir,
        signal_columns=signal_columns,
        device=0,
        seed=seed,
    )

    experiments.parallel_run(
        max_workers=90,
        sleep_time=0.2,
        retrain=True,
        reembed=True,
        recompute_mutual_information=True,
    )
