from scaling_laws.prepare.data import Experiments
import numpy as np

import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

src_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19_48_download_necessary_models"
dst_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19-54_evaluate_mid_quality_models_on_full_data"

if not os.path.exists(dst_folder):
    shutil.copytree(src_folder, dst_folder)

experiment_configs = {
    "PBMC": (100000, 0.1072766, ["celltype.l3", "protein_counts"]),
    # "larry": (100000, 0.0847557, ["index", "clone", "time"]),
    # "shendure": (10000000, 0.0859506, ["author_day"]),
    # "merfish": (60000, 0.0905502, ["cur_idx", "ng_idx"]),
}

def run_dataset_experiments(dataset_name, size, quality, signal_columns, dst_folder):
    """Run experiments for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Running experiments for {dataset_name}: size={size}, quality={quality}")
    print(f"Signal columns: {signal_columns}")
    print(f"{'='*80}\n")
    
    experiments: Experiments = Experiments(
        datasets=[dataset_name],
        sizes=[size],
        qualities=[quality],
        # algos=["Geneformer", "PCA", "SCVI", "RandomProjection"],
        algos=["Geneformer"],
        path_to_data_dir=dst_folder,
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
    
    print(f"\nCompleted experiments for {dataset_name}\n")
    return dataset_name

# Run all datasets in parallel
with ThreadPoolExecutor(max_workers=len(experiment_configs)) as executor:
    futures = {
        executor.submit(
            run_dataset_experiments, 
            dataset_name, 
            size, 
            quality, 
            signal_columns, 
            dst_folder
        ): dataset_name
        for dataset_name, (size, quality, signal_columns) in experiment_configs.items()
    }
    
    # Wait for all to complete and handle results
    for future in as_completed(futures):
        dataset_name = futures[future]
        try:
            result = future.result()
            print(f"✓ Successfully completed {result}")
        except Exception as e:
            print(f"✗ Error running experiments for {dataset_name}: {e}")

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)
