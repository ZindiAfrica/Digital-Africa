#!/bin/bash

set -e

ENV_NAME="ivory-env"
PYTHON_VERSION="3.9"
REQUIREMENTS_FILE="requirements.txt"

# Step 1: Install Miniconda if not found
if ! command -v conda &> /dev/null; then
    echo "📥 Miniconda not found. Installing..."
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    echo "✅ Miniconda installed."
else
    echo "✅ Conda is already installed."
    eval "$(conda shell.bash hook)"
fi

# Step 2: Create Conda env
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "🔁 Activating existing environment: $ENV_NAME"
else
    echo "🆕 Creating new Conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# Step 3: Activate and install packages
conda activate $ENV_NAME
echo "📦 Installing packages from $REQUIREMENTS_FILE..."
pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE

echo "✅ Conda environment '$ENV_NAME' is ready!"
