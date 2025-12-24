#!/bin/bash
# submit_kidney_experiments.sh

DATASET="tissuemnist"
OUTPUT_DIR="tissuemnist_models"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p slurm_logs

# Submit clean experiment
# sbatch run_single_experiment.slurm clean none

# Submit Gaussian noise experiments
for sigma in 0.01 0.027826 0.077426 0.215443 0.599484 1.6681 4.64159 12.9155 35.9381 100.0; do
    sbatch run_single_experiment.slurm gauss ${sigma}
done

# Submit pixelation experiments
for scale in 2 4 7 8 14 16 28 32 56 112 224; do
# for scale in 2 4 7 14 28; do
    sbatch run_single_experiment.slurm pix ${scale}
done

echo "All jobs submitted! Check status with: squeue -u \$USER"