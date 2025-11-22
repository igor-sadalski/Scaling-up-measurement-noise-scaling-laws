import anndata as ad
import scanpy as sc
import pickle
import os
import numpy as np
from pathlib import Path


def prepare_hvg(
    input_h5ad_path: str,
    output_dir: str,
    n_top_genes: int = 750,
    max_cells: int | None = None,
    seed: int = 0,
    target_sum: float = 1e4,
    output_filename: str = "pca_hvg.pkl",
):
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing: {input_h5ad_path}")
    if os.path.exists(output_path):
        print(f"  Note: Output file already exists, will be overwritten: {output_path}")
    
    try:
        print(f"Step 1/6: Reading h5ad file...")
        adata = ad.read_h5ad(input_h5ad_path)
        print(f"  Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    except Exception as e:
        raise Exception(f"Failed to read h5ad file: {e}")
    
    try:
        if max_cells is not None and adata.n_obs > max_cells:
            print(f"Step 2/6: Subsample {max_cells} cells from {adata.n_obs} total cells...")
            np.random.seed(seed)
            idx = np.random.choice(adata.n_obs, max_cells, replace=False)
            adata = adata[idx]
            print(f"  Subsample complete: {adata.n_obs} cells")
        else:
            print(f"Step 2/6: No subsampling needed (using all {adata.n_obs} cells)")
    except Exception as e:
        raise Exception(f"Failed to subsample cells: {e}")
    
    try:
        print(f"Step 3/6: Normalizing and log transforming...")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print(f"  Normalization complete")
    except Exception as e:
        raise Exception(f"Failed to normalize and log transform: {e}")
    
    try:
        print(f"Step 4/6: Finding {n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_mask = adata.var["highly_variable"].to_numpy()
        n_hvg = hvg_mask.sum()
        print(f"  Found {n_hvg} highly variable genes")
    except Exception as e:
        raise Exception(f"Failed to find highly variable genes: {e}")
    
    try:
        print(f"Step 5/6: Saving HVG mask to file...")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(hvg_mask, f)
        print(f"Step 6/6: Complete!")
        print(f"✓ Successfully created/updated output file: {output_path}\n")
    except Exception as e:
        raise Exception(f"Failed to save HVG mask: {e}")
    
    return hvg_mask


if __name__ == "__main__":
    print("=" * 70)
    print("Starting HVG preparation process")
    print("=" * 70 + "\n")
    
    datasets = [
        {
            "input_h5ad_path": "/mnt/nvme/noise_laws/data/larry/100000/1.0/preprocessed/preprocessed.h5ad",
            "output_dir": "/mnt/nvme/noise_laws/data/larry/utils",
            "n_top_genes": 750,
            "max_cells": 300_000,
            "seed": 0,
        },
        {
            "input_h5ad_path": "/mnt/nvme/noise_laws/data/shendure/10000000/1.0/preprocessed/preprocessed.h5ad",
            "output_dir": "/mnt/nvme/noise_laws/data/shendure/utils",
            "n_top_genes": 750,
        },
        {
            "input_h5ad_path": "/mnt/nvme/noise_laws/data/merfish/100000/1.0/preprocessed/preprocessed.h5ad",
            "output_dir": "/mnt/nvme/noise_laws/data/merfish/utils",
            "n_top_genes": 750,
        },
        {
            "input_h5ad_path": "/mnt/nvme/noise_laws/data/PBMC/100000/1.0/preprocessed/preprocessed.h5ad",
            "output_dir": "/mnt/nvme/noise_laws/data/PBMC/utils",
            "n_top_genes": 750,
        },
    ]
    
    successful = []
    failed = []
    
    for i, dataset in enumerate(datasets, 1):
        print(f"[Dataset {i}/{len(datasets)}]")
        dataset_name = dataset.get("input_h5ad_path", f"Dataset {i}")
        try:
            prepare_hvg(**dataset)
            successful.append(dataset_name)
        except Exception as e:
            print(f"✗ ERROR: Failed to process {dataset_name}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            print()
            failed.append((dataset_name, str(e)))
    
    print("=" * 70)
    print("HVG preparation process completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Successful: {len(successful)}/{len(datasets)}")
    if successful:
        for name in successful:
            print(f"    ✓ {name}")
    if failed:
        print(f"  Failed: {len(failed)}/{len(datasets)}")
        for name, error in failed:
            print(f"    ✗ {name}")
            print(f"      Error: {error}")
    print("=" * 70)