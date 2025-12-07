#!/usr/bin/env python3

import sys
from pathlib import Path

scaling_laws_path = Path(__file__).parent.parent / "scaling_laws" / "src"
sys.path.insert(0, str(scaling_laws_path))

from scaling_laws.algo import Geneformer, PCA, SCVI, RandomProjection
import os
import shutil
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import torch
from tqdm import tqdm

RESULTS_BASE = Path("/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19_48_download_necessary_models")
OUTPUT_BASE = Path("/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19-54_evaluate_mid_quality_models_on_full_data")

ALGO_CLASSES = {
    "Geneformer": Geneformer,
    "PCA": PCA,
    "SCVI": SCVI,
    "RandomProjection": RandomProjection,
}


def find_trained_models(results_dir: Path, algo_name: str) -> list[Path]:
    algo_dir = results_dir / algo_name
    if not algo_dir.exists():
        return []
    
    models = []
    
    if algo_name == "Geneformer":
        model_dir = algo_dir / "model"
        if model_dir.exists() and (model_dir / "model.safetensors").exists():
            models.append(model_dir)
    else:
        model_dir = algo_dir / "model"
        if model_dir.exists():
            if algo_name == "PCA" and (model_dir / "pca_model.pkl").exists():
                models.append(model_dir)
            elif algo_name == "SCVI" and (any(model_dir.glob("*.pt")) or any(model_dir.glob("*.pkl")) or (model_dir / "model.pt").exists()):
                models.append(model_dir)
            elif algo_name == "RandomProjection" and (model_dir / "random_projection.joblib").exists():
                models.append(model_dir)
    
    return models


def setup_temp_base_dir(dataset: str, size: str, quality: str, test_data_path: Path, results_base: Path) -> Path:
    temp_base = OUTPUT_BASE / "temp" / dataset / size / quality
    temp_base.mkdir(parents=True, exist_ok=True)
    
    preprocessed_dir = temp_base / "preprocessed"
    if preprocessed_dir.exists():
        if preprocessed_dir.is_symlink():
            preprocessed_dir.unlink()
        elif preprocessed_dir.is_dir():
            shutil.rmtree(preprocessed_dir)
    preprocessed_dir.symlink_to(test_data_path)
    
    test_dir = OUTPUT_BASE / "temp" / dataset / "test" / quality
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_preprocessed = test_dir / "preprocessed"
    if test_preprocessed.exists():
        if test_preprocessed.is_symlink():
            test_preprocessed.unlink()
        elif test_preprocessed.is_dir():
            shutil.rmtree(test_preprocessed)
    test_preprocessed.symlink_to(test_data_path)
    
    tokenized_dataset = test_preprocessed / "tokenized.dataset"
    if not tokenized_dataset.exists():
        print(f"Warning: tokenized.dataset not found at {tokenized_dataset} (may be needed for Geneformer)")
    
    signals_source = results_base / dataset / "test" / "1.0" / "signals"
    if signals_source.exists():
        signals_dir = test_dir / "signals"
        if signals_dir.exists():
            if signals_dir.is_symlink():
                signals_dir.unlink()
            elif signals_dir.is_dir():
                shutil.rmtree(signals_dir)
        signals_dir.symlink_to(signals_source)
    
    utils_dir = OUTPUT_BASE / "temp" / dataset / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    utils_source_dir = results_base / dataset / "utils"
    if not utils_source_dir.exists():
        utils_source_dir = RESULTS_BASE / dataset / "utils"
    
    if utils_source_dir.exists() and utils_source_dir.is_dir():
        utils_files = [
            "token_dict.pkl",
            "pca_hvg.pkl",
            "gene_median_dict.pkl",
            "detected_gene_median_dict.pkl",
            "ensembl_mapping_dict.pkl",
            "total_gene_tdigest_dict.pkl",
        ]
        
        for utils_file in utils_files:
            source_file = utils_source_dir / utils_file
            if source_file.exists():
                target_file = utils_dir / utils_file
                if target_file.exists():
                    if target_file.is_symlink():
                        target_file.unlink()
                    else:
                        target_file.unlink()
                target_file.symlink_to(source_file)
                print(f"Linked {utils_file} to {target_file}")
    else:
        print(f"Warning: Utils directory not found at {utils_source_dir}")
    
    return temp_base


def evaluate_model(
    model_path: Path,
    algo_name: str,
    dataset: str,
    size: str,
    quality: str,
    test_data_path: Path,
    device: int = 0,
    seed: int = 42,
    inference_batch_size: Optional[int] = None,
) -> dict:
    print(f"\n{'='*80}")
    print(f"Evaluating {algo_name} model: {model_path.name}")
    print(f"Dataset: {dataset}, Size: {size}, Quality: {quality}")
    print(f"{'='*80}\n")
    
    temp_base = setup_temp_base_dir(dataset, size, quality, test_data_path, RESULTS_BASE)
    
    try:
        if algo_name == "Geneformer" and model_path.name.startswith("checkpoint-"):
            model_name = model_path.name
        else:
            model_name = "model"
        
        algo_kwargs = {
            "base_dir": str(temp_base),
            "device": device,
            "seed": seed,
        }
        
        signal_columns_map = {
            "larry": ["clone"],
            "merfish": ["ng_idx"],
            "PBMC": ["celltype.l3", "protein_counts"],
            "shendure": ["author_day"],
        }
        signal_columns = signal_columns_map.get(dataset, ["celltype.l3"])
        
        if algo_name == "Geneformer":
            algo_kwargs["model_name"] = model_name
            algo_kwargs["dataset_name"] = dataset
            algo_kwargs["signal_columns"] = signal_columns
        elif algo_name == "SCVI":
            algo_kwargs["dataset_name"] = dataset
            algo_kwargs["signal_columns"] = signal_columns
        elif algo_name == "PCA":
            algo_kwargs["signal_columns"] = signal_columns
        
        algo_class = ALGO_CLASSES[algo_name]
        algo = algo_class(**algo_kwargs)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        if algo_name == "Geneformer":
            target_model_path = algo.model_path
            if target_model_path.exists():
                if target_model_path.is_symlink():
                    target_model_path.unlink()
                elif target_model_path.is_dir():
                    shutil.rmtree(target_model_path)
                else:
                    target_model_path.unlink()
            target_model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.is_dir():
                if (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists():
                    shutil.copytree(model_path, target_model_path)
                else:
                    shutil.copytree(model_path, target_model_path)
            else:
                target_model_path.symlink_to(model_path)
        else:
            target_model_path = algo.model_path
            if target_model_path.exists():
                shutil.rmtree(target_model_path)
            target_model_path.mkdir(parents=True, exist_ok=True)
            
            if algo_name == "PCA":
                pca_file = model_path / "pca_model.pkl"
                if not pca_file.exists():
                    raise FileNotFoundError(f"PCA model file not found: {pca_file}")
                shutil.copy(pca_file, target_model_path / "pca_model.pkl")
            elif algo_name == "SCVI":
                for item in model_path.iterdir():
                    if item.is_file():
                        shutil.copy(item, target_model_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, target_model_path / item.name)
            elif algo_name == "RandomProjection":
                rp_file = model_path / "random_projection.joblib"
                if not rp_file.exists():
                    raise FileNotFoundError(f"RandomProjection model file not found: {rp_file}")
                shutil.copy(rp_file, target_model_path / "random_projection.joblib")
        
        print(f"Running inference with {algo_name}...")
        if algo_name == "Geneformer":
            embeddings = algo.embed(inference_batch_size=inference_batch_size or 50)
        else:
            embeddings = algo.embed()
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        output_dir = OUTPUT_BASE / dataset / size / quality / algo_name / model_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_output = output_dir / "embeddings.csv"
        pd.DataFrame(embeddings).to_csv(embeddings_output, index=False)
        print(f"Saved embeddings to {embeddings_output}")
        
        algo.embeddings_path = embeddings_output
        
        print(f"Computing LMI for {algo_name}...")
        mi_results = algo.mutual_information(max_epochs=300)
        
        mi_source_dir = algo.save_folder_path / algo.model_path.name / "MI"
        if mi_source_dir.exists():
            mi_target_dir = output_dir / "MI"
            if mi_target_dir.exists():
                shutil.rmtree(mi_target_dir)
            shutil.copytree(mi_source_dir, mi_target_dir)
            print(f"Copied LMI results to {mi_target_dir}")
        else:
            print(f"Warning: LMI results directory not found at {mi_source_dir}")
        
        mi_summary = output_dir / "lmi_summary.txt"
        with open(mi_summary, "w") as f:
            f.write("LMI Results:\n")
            f.write("=" * 50 + "\n")
            for signal_file, mi_value in mi_results.items():
                f.write(f"{signal_file}: {mi_value:.5f}\n")
        
        print(f"LMI computation complete. Results saved to {output_dir}")
        print(f"MI Results: {mi_results}")
        
        return {
            "dataset": dataset,
            "size": size,
            "quality": quality,
            "algorithm": algo_name,
            "model_name": model_path.name,
            "mi_results": mi_results,
            "status": "success",
        }
        
    except Exception as e:
        print(f"Error evaluating {algo_name} model {model_path.name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "dataset": dataset,
            "size": size,
            "quality": quality,
            "algorithm": algo_name,
            "model_name": model_path.name,
            "mi_results": {},
            "status": "failed",
            "error": str(e),
        }
    finally:
        pass


def save_results_to_csv(all_results: list[dict]) -> None:
    rows = []
    
    for result in all_results:
        base_row = {
            "dataset": result["dataset"],
            "size": result["size"],
            "quality": result["quality"],
            "algorithm": result["algorithm"],
            "model_name": result["model_name"],
            "status": result["status"],
        }
        
        if result["status"] == "failed":
            base_row["error"] = result.get("error", "Unknown error")
        
        mi_results = result.get("mi_results", {})
        if mi_results:
            for signal_file, mi_value in mi_results.items():
                row = base_row.copy()
                row["signal_file"] = signal_file
                row["lmi"] = mi_value
                rows.append(row)
        else:
            row = base_row.copy()
            row["signal_file"] = None
            row["lmi"] = None
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    column_order = ["dataset", "size", "quality", "algorithm", "model_name", "signal_file", "lmi", "status"]
    if "error" in df.columns:
        column_order.append("error")
    
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    csv_path = OUTPUT_BASE / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    
    summary_rows = []
    for result in all_results:
        row = {
            "dataset": result["dataset"],
            "size": result["size"],
            "quality": result["quality"],
            "algorithm": result["algorithm"],
            "model_name": result["model_name"],
            "status": result["status"],
        }
        
        if result["status"] == "failed":
            row["error"] = result.get("error", "Unknown error")
        
        mi_results = result.get("mi_results", {})
        for signal_file, mi_value in mi_results.items():
            col_name = signal_file.replace(".csv", "").replace("Y_", "lmi_")
            row[col_name] = mi_value
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_BASE / "evaluation_results_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Total models: {len(summary_df)}")


def evaluate_model_wrapper(args):
    (model_path, algo_name, dataset, size, quality, test_data_path, 
     device, seed, inference_batch_size) = args
    
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    test_data_path = Path(test_data_path) if isinstance(test_data_path, str) else test_data_path
    
    return evaluate_model(
        model_path=model_path,
        algo_name=algo_name,
        dataset=dataset,
        size=size,
        quality=quality,
        test_data_path=test_data_path,
        device=device,
        seed=seed,
        inference_batch_size=inference_batch_size,
    )


def main():
    print(f"Results base: {RESULTS_BASE}")
    print(f"Output base: {OUTPUT_BASE}")
    
    tasks = []
    
    dataset_dirs = [d for d in RESULTS_BASE.iterdir() if d.is_dir() and d.name != "temp"]
    
    print(f"\nCollecting tasks from {len(dataset_dirs)} dataset(s)...")
    
    for dataset_dir in tqdm(dataset_dirs, desc="Scanning datasets", unit="dataset"):
        dataset = dataset_dir.name
        
        test_data_path = dataset_dir / "test" / "1.0" / "preprocessed"
        if not test_data_path.exists():
            print(f"Warning: Test data not found at {test_data_path}")
            continue
        
        for size_dir in dataset_dir.iterdir():
            if not size_dir.is_dir() or size_dir.name in ["test", "utils", "raw"]:
                continue
            
            size = size_dir.name
            
            for quality_dir in size_dir.iterdir():
                if not quality_dir.is_dir():
                    continue
                
                quality = quality_dir.name
                
                results_dir = quality_dir / "results"
                if not results_dir.exists():
                    results_dir = quality_dir
                    if not any(results_dir.glob("Geneformer")) and not any(results_dir.glob("PCA")):
                        continue
                
                for algo_name in ALGO_CLASSES.keys():
                    algo_dir = results_dir / algo_name
                    if not algo_dir.exists():
                        continue
                    
                    models = find_trained_models(results_dir, algo_name)
                    
                    for model_idx, model_path in enumerate(models):
                        if torch.cuda.is_available():
                            num_devices = torch.cuda.device_count()
                            device = model_idx % num_devices
                        else:
                            device = 0
                        
                        tasks.append((
                            str(model_path),
                            algo_name,
                            dataset,
                            size,
                            quality,
                            str(test_data_path),
                            device,
                            42,
                            50 if algo_name == "Geneformer" else None,
                        ))
    
    print(f"\n\nTotal tasks collected: {len(tasks)}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_workers = min(len(tasks), num_gpus * 2)
        print(f"Detected {num_gpus} GPU(s), using {max_workers} workers")
    else:
        max_workers = min(len(tasks), multiprocessing.cpu_count())
        print(f"Using {max_workers} CPU workers")
    
    print(f"Running evaluations in parallel...\n")
    
    completed = 0
    failed = 0
    all_results = []
    
    pbar = tqdm(total=len(tasks), desc="Evaluating models", unit="model", ncols=100)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(evaluate_model_wrapper, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            model_path, algo_name, dataset, size, quality = task[0], task[1], task[2], task[3], task[4]
            model_name = Path(model_path).name if isinstance(model_path, str) else model_path.name
            
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                completed += 1
                
                pbar.set_postfix({
                    "Completed": completed,
                    "Failed": failed,
                    "Current": f"{algo_name}/{model_name[:20]}"
                })
                pbar.update(1)
            except Exception as e:
                failed += 1
                print(f"\n[Error] Failed to evaluate {algo_name} model {model_name} for {dataset}/{size}/{quality}: {e}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    "dataset": dataset,
                    "size": size,
                    "quality": quality,
                    "algorithm": algo_name,
                    "model_name": model_name,
                    "mi_results": {},
                    "status": "failed",
                    "error": str(e),
                })
                
                pbar.set_postfix({
                    "Completed": completed,
                    "Failed": failed,
                    "Current": f"{algo_name}/{model_name[:20]} (FAILED)"
                })
                pbar.update(1)
    
    pbar.close()
    
    print(f"\n\n{'='*80}")
    print(f"All evaluations complete!")
    print(f"Total tasks: {len(tasks)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"{'='*80}")
    
    print(f"\nSaving results to CSV...")
    save_results_to_csv(all_results)
    print(f"Results saved to CSV file.")


if __name__ == "__main__":
    main()

