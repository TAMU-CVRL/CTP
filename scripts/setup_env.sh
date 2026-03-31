#!/bin/bash

set -e
ENV_NAME="ctp"

echo "Starting to configure the $ENV_NAME environment..."

eval "$(conda shell.bash hook)"

# Check if the conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Activating and continuing installation..."
else
    echo "Creating conda environment: $ENV_NAME (Python 3.10)..."
    # -y automatically answers 'yes' to the installation prompts
    conda create -y -n $ENV_NAME python=3.10
fi

# Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch
echo "Installing PyTorch (CUDA 12.6), please refer PyTorch official website for specific GPUs..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other core dependencies
echo "Installing other required libraries..."
pip install transformers==5.1.0 \
            nuscenes-devkit==1.2.0 \
            pandas==2.3.3 \
            open3d==0.19.0 \
            wandb==0.25.1 \
            tensorboard==2.20.0 \
            git+https://github.com/openai/CLIP.git \
            matplotlib==3.9.4 \
            huggingface_hub==1.8.0 \
            umap-learn==0.5.11 \
            accelerate==1.13.0 \
            beautifulsoup4==4.14.3 \
            typeguard==4.4.4 \
            pyyaml==6.0.3 \
            tqdm==4.67.1 \
            idna==3.11 \
            ipykernel==7.2.0 \
            ipywidgets==8.1.8 \
            pickleshare==0.7.5 \
            jmespath==1.1.0 \
            pyrootutils==1.0.4 \

echo "Environment setup is complete!"
echo "Please run the following command in your terminal to activate and use it:"
echo "conda activate $ENV_NAME"