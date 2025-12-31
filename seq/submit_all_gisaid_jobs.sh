# seq/submit_all_gisaid_jobs.sh
#!/bin/bash
# Submit all GISAID training jobs

# Create output directories
mkdir -p seq/slurm_logs
mkdir -p seq

# Define model configurations
declare -A models
models["8M"]="facebook/esm2_t6_8M_UR50D"
models["35M"]="facebook/esm2_t12_35M_UR50D"
models["150M"]="facebook/esm2_t30_150M_UR50D"

# noise levels (must match what was generated in data prep!)
NOISE_LEVELS=(0.999001 0.990099 0.909091 0.500000 0.090909 0.009901 0.000999 0.000100 0.000010 0.000001)


echo "Submitting jobs for all model sizes and noise levels..."

for model_name in "${!models[@]}"; do
    model_id="${models[$model_name]}"
    echo "Model: $model_name ($model_id)"
    
    for noise_level in "${NOISE_LEVELS[@]}"; do
        data_path="seq/data_noise_${noise_level}.pt"
        
        if [ -f "$data_path" ]; then
            echo "  Submitting: noise=$noise_level"
            sbatch seq/run_gisaid_training.slurm "$model_name" "$model_id" "$noise_level" "$data_path"
        else
            echo "  WARNING: Data file not found: $data_path"
        fi
    done
done

echo ""
echo "All jobs submitted!"
echo "Monitor logs in: seq/slurm_logs/"