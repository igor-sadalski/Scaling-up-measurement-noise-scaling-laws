#!/bin/bash
#SBATCH --job-name=kidney_exp
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# run_single_experiment.slurm

# Get arguments
NOISE_TYPE=$1
NOISE_PARAM=$2

# Load necessary modules
# module load python/3.9
# module load cuda/11.8

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate scaling

# Run the experiment
if [ "$NOISE_TYPE" == "clean" ]; then
    python run_kidney_experiment.py \
        --noise-type clean \
        --dataset tissuemnist \
        --output-dir tissuemnist_models \
        --epochs 30 \
        --size 224
else
    python run_kidney_experiment.py \
        --noise-type ${NOISE_TYPE} \
        --noise-param ${NOISE_PARAM} \
        --dataset tissuemnist \
        --output-dir tissuemnist_models \
        --epochs 30 \
        --size 224
fi