from scaling_laws.prepare.data import Experiments
import numpy as np
from pathlib import Path

import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

src_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-12_download_different_geneformer_models"
dst_folder = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-21_evaluate_all_shendure_geneformer_models_on_mid_quality_data"

if not os.path.exists(dst_folder):
    shutil.copytree(src_folder, dst_folder)

dataset_name = "shendure"
size = 10000000
signal_columns = ["author_day"]

qualities = [1.0, 0.5414548, 0.2931733, 0.1587401, 0.0859506, 0.0465384, 0.0251984, 0.0136438, 0.0073875, 0.004]

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
    
    checkpoint_dirs = [p for p in model_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint")]
    if not checkpoint_dirs:
        print(f"Warning: No checkpoints found in {model_path}")
        return None
    
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda p: int(p.name.split("-")[1]) if "-" in p.name and p.name.split("-")[1].isdigit() else 0,
    )
    checkpoint_path = checkpoint_dirs[-1]
    
    print(f"Using checkpoint: {checkpoint_path}")
    
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
        max_workers=100,
        sleep_time=0.2,
        retrain=False,
        reembed=True,
        recompute_mutual_information=True,
        reembed_checkpoint=str(checkpoint_path),
        mem_limit={"Geneformer": 23_000, "default": 23_000},
    )
    
    print(f"\nCompleted evaluation for model trained on quality {quality}\n")
    return quality

with ThreadPoolExecutor(max_workers=len(qualities)) as executor:
    futures = {
        executor.submit(
            run_model_evaluation, 
            quality,
            dataset_name, 
            size, 
            signal_columns, 
            dst_folder
        ): quality
        for quality in qualities
    }
    
    for future in as_completed(futures):
        quality = futures[future]
        try:
            result = future.result()
            if result:
                print(f"✓ Successfully completed evaluation for model trained on quality {result}")
        except Exception as e:
            print(f"✗ Error evaluating model trained on quality {quality}: {e}")

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)
