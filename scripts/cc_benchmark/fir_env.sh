#!/bin/bash
# FIR cluster environment setup

# Load Python module (adjust version if needed)
module load python/3.11
module load scipy-stack

# Load CUDA and cuDNN for JAX GPU support
module load cuda/12.2
module load cudnn

# Export path for XLA to find CUDA
if [ -n "$CUDA_HOME" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi

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
    
    # Check if we are in a GPU job or have CUDA loaded
    if [[ "$SLURM_JOB_PARTITION" == *"gpu"* ]] || [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "GPU job detected, installing with [all,gpu] extras..."
        # On CC, we use the extras from pyproject.toml but point to the JAX releases URL
        pip install -e "../../[all,gpu]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "CPU job detected, installing with [all] extra..."
        pip install -e "../../[all]"
    fi
    
    # Install additional benchmarking dependencies
    pip install mne-bids pandas openneuro-py
    
    echo "Environment installed. JAX version:"
    python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
else
    echo "Activating existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi
