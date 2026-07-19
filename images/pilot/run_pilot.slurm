#!/bin/bash
#SBATCH --job-name=batch_effect_pilot
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# run_pilot.slurm -- multi-seed convenience wrapper.
# NOTE: you can also submit the python script directly (it carries its own #SBATCH header
# and auto-runs the analysis at the end):
#     sbatch batch_effect_experiment.py
#
# This wrapper just overrides a few args (e.g. 3 seeds, 30 epochs) and uses the `lt` env.

mkdir -p slurm_logs

eval "$(conda shell.bash hook)"
conda activate lt

cd "$(dirname "$0")"

# the script builds figures + summary.json itself (drop --no-analyze to keep that)
python batch_effect_experiment.py \
    --dataset tissuemnist \
    --size 224 \
    --epochs 30 \
    --k 100 \
    --seeds 0 1 2 \
    --output-dir results
