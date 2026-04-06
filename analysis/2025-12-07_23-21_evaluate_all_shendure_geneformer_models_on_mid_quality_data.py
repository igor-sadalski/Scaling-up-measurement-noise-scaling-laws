from scaling_laws.prepare.data import Experiments
import numpy as np
from pathlib import Path

import shutil
import os

src_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-12_download_different_geneformer_models"
dst_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-21_evaluate_all_shendure_geneformer_models_on_mid_quality_data"

if not os.path.exists(dst_folder):
    shutil.copytree(src_folder, dst_folder)

# qualities = [1.0, 0.5414548, 0.2931733, 0.1587401, 0.0859506, 0.0465384, 0.0251984, 0.0136438, 0.0073875, 0.004]
qualities = [0.0251984]

datasets = {
    "shendure": (10000000, qualities, ["author_day"]),
}

def run_model_evaluation(quality, dataset_name, size, signal_columns, dst_folder):
    """Evaluate a model trained on quality on the same quality data."""
    print(f"\n{'='*80}")
    print(f"Evaluating model trained on quality {quality} on quality {quality} data")
    print(f"Dataset: {dataset_name}, Size: {size}")
    print(f"{'='*80}\n")
    
    model_path = Path(dst_folder) / dataset_name / str(size) / str(quality) / "results" / "Geneformer"
    
    if not model_path.exists():
        print(f"Warning: Model path does not exist: {model_path}")
        return None
    
    # Use the model directory directly instead of checkpoints
    model_dir = model_path / "model"
    
    if not model_dir.exists():
        print(f"Warning: Model directory does not exist: {model_dir}")
        return None
    
    print(f"Using model: {model_dir}")
    
    experiments: Experiments = Experiments(
        datasets=[dataset_name],
        sizes=[size],
        qualities=[quality],
        algos=["Geneformer"],
        path_to_data_dir=dst_folder,
        signal_columns=signal_columns,
        device=0,
        seed=42,
    )
    
    experiments.parallel_run(
        max_workers=7,
        sleep_time=0.2,
        retrain=False,
        reembed=True,
        recompute_mutual_information=True,
        reembed_checkpoint=str(model_dir),
        mem_limit={"Geneformer": 23_000, "default": 23_000},
    )
    
    print(f"\nCompleted evaluation for model trained on quality {quality}\n")
    return quality

# Iterate over datasets sequentially
for dataset_name, (size, qualities_list, signal_columns) in datasets.items():
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Size: {size}, Qualities: {len(qualities_list)}, Signal columns: {signal_columns}")
    print(f"{'='*80}\n")
    
    # Iterate over qualities sequentially
    for quality in qualities_list:
        try:
            result = run_model_evaluation(
                quality,
                dataset_name,
                size,
                signal_columns,
                dst_folder
            )
            if result:
                print(f"✓ Successfully completed evaluation for model trained on quality {result}")
        except Exception as e:
            print(f"✗ Error evaluating model trained on quality {quality}: {e}")

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)
