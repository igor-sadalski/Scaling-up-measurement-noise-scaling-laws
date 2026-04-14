from scaling_laws.prepare.data import Experiments
import numpy as np

datasets = ["merfish"]
sizes = list(map(int, np.logspace(np.log10(60000), 2, 10)))
qualities = list(map(lambda x: round(x, 7), np.logspace(0, np.log10(10 / 367), 10)))
path_to_data_dir = "/home/igor/noise_scaling/data"
signal_columns = ["cur_idx", "ng_idx"]
seeds = [42]

experiments: Experiments = Experiments(
    datasets=datasets,
    sizes=sizes,
    qualities=qualities,
    algos=["State"],
    path_to_data_dir=path_to_data_dir,
    signal_columns=signal_columns,
    seed=seeds[0],
)

experiments.prepare_state_data(max_workers=8)

for seed in seeds:

    experiments: Experiments = Experiments(
        datasets=datasets,
        sizes=sizes,
        qualities=qualities,
        algos=["State"],
        path_to_data_dir=path_to_data_dir,
        signal_columns=signal_columns,
        device=0,
        seed=seed,
    )

    experiments.parallel_run(
        max_workers=8,
        sleep_time=0.2,
        retrain=True,
        reembed=True,
        recompute_mutual_information=True,
        max_epochs=10,
        early_stopping_patience=5,
        jobs_per_gpu=1,
        log_dir=f"{path_to_data_dir}/merfish/logs/state_run_seed_{seed}",
    )
