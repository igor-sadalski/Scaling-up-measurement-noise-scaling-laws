#!/bin/bash
# Script to install dependencies for the evaluation script

# Activate conda environment (adjust name if needed)
ENV_NAME="noise_scaling"

echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment $ENV_NAME"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Install packages via conda (preferred for scientific packages)
echo ""
echo "Installing packages via conda..."
conda install -y -c conda-forge -c bioconda \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    joblib \
    anndata \
    scanpy \
    pytorch \
    pytorch-cuda=12.1 \
    -c pytorch

# Install packages via pip that aren't available in conda
echo ""
echo "Installing packages via pip..."
python -m pip install --upgrade pip

# Core dependencies
python -m pip install \
    latentmi \
    scvi-tools \
    transformers==4.40 \
    datasets \
    wandb \
    pytorch-lightning

# Install geneformer if available
if [ -d "Geneformer" ]; then
    echo ""
    echo "Installing Geneformer..."
    cd Geneformer
    python -m pip install -e .
    cd ..
fi

# Install scaling_laws package in development mode
echo ""
echo "Installing scaling_laws package..."
cd scaling_laws
python -m pip install -e .
cd ..

echo ""
echo "Installation complete!"
echo "To verify, try: python -c 'import torch; import scvi; import geneformer; print(\"All packages imported successfully\")'"




