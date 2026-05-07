#!/bin/bash
# FIR cluster environment setup

# Load Python module (adjust version if needed)
module load python/3.11
module load scipy-stack

# Load CUDA if GPU is needed (JAX will detect it)
module load cuda/12.2

if [ -n "$SLURM_TMPDIR" ]; then
    # Running on cluster node: use local SSD for speed
    VENV_PATH="$SLURM_TMPDIR/venv"
    REINSTALL=true
else
    # Running locally: use persistent venv
    VENV_PATH="./.venv_fir"
    REINSTALL=false
    if [ ! -d "$VENV_PATH" ]; then REINSTALL=true; fi
fi

if [ "$REINSTALL" = true ]; then
    echo "Setting up virtual environment at $VENV_PATH..."
    virtualenv --no-download "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --no-index --upgrade pip
    
    # Install core dependencies from local checkout
    # Note: using --no-index where possible, but may need internet for some
    pip install -e "../../[all]"
    
    # Install additional benchmarking dependencies
    pip install mne-icalabel mne-bids pandas openneuro-py
else
    echo "Activating existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Set JAX to use GPU by default if available, or CPU
# export JAX_PLATFORM_NAME=gpu
