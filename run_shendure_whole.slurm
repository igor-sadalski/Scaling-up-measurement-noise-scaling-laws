#!/bin/bash
#SBATCH --job-name=shendure_whole
#SBATCH --output=/home/igor/exploration/Scaling-up-measurement-noise-scaling-laws/output/shendure_whole_%j.out
#SBATCH --error=/home/igor/exploration/Scaling-up-measurement-noise-scaling-laws/output/shendure_whole_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=mi325x-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=200G
#SBATCH --gres=gpu:1

# Change to the script directory
cd /home/igor/exploration/Scaling-up-measurement-noise-scaling-laws

# Activate conda environment if needed (uncomment and modify if necessary)
# source activate your_env_name

# Run the Python script
python run_shendure_whole.py

