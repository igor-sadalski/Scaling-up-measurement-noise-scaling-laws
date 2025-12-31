#!/bin/bash
#SBATCH --job-name=gisaid_train
#SBATCH --output=seq/slurm_logs/%x_%A_%a.out
#SBATCH --error=seq/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Get arguments
MODEL_NAME=$1
MODEL_ID=$2
NOISE_LEVEL=$3
DATA_PATH=$4

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate scaling

# Run training
python seq/train_gisaid_model.py \
    --model-name ${MODEL_NAME} \
    --model-id ${MODEL_ID} \
    --noise-level ${NOISE_LEVEL} \
    --data-path ${DATA_PATH} \
    --output-dir seq

echo "Job completed!"