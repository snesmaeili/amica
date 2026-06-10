#!/bin/bash
# FIR cluster environment setup for AMICA benchmarking
# Ref: https://github.com/BabaSanfour/crash-course/tree/main/module_05_advanced_alliance_ai_workflows

# ── Module loads ──
# Force all caches to scratch to avoid /home quota Input/output errors
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export XDG_DATA_HOME="/scratch/$USER/.local/share"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$PIP_CACHE_DIR"

module purge
module load StdEnv/2023 || true
module load python/3.11
module load scipy-stack

# Load CUDA and cuDNN for JAX GPU support
module load cuda/12.6
module load cudnn

# Export path for XLA to find CUDA
if [ -n "$CUDA_HOME" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi

# ── Dataset paths (crash-course rule: reuse shared /project datasets) ──
export BIDS_ROOT_DS4505="/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids"

# ── NumPy/MKL thread tuning ──
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ── Results directory ──
# Temporarily pointing to scratch because the team /project quota is full!
RESULTS_DIR="/scratch/$USER/amica_python_validation_outputs"
export AMICA_RESULTS_DIR="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ── Virtual environment ──
# Use a persistent venv in the repo root (created on the login node with internet)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv_fir"

if [ ! -d "$VENV_PATH" ]; then
    REINSTALL=true
else
    REINSTALL=false
fi



if [ "$REINSTALL" = true ]; then
    echo "Setting up virtual environment at $VENV_PATH..."
    virtualenv --no-download "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --no-index --upgrade pip

    # Check if we are in a GPU job
    if [[ "$SLURM_JOB_PARTITION" == *"gpu"* ]] || [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "GPU job detected, installing with [all,gpu] extras..."
        pip install -e "$REPO_ROOT[all,gpu]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "CPU job detected, installing with [all] extra..."
        pip install -e "$REPO_ROOT[all]"
    fi

    # Install additional benchmarking dependencies
    pip install mne-bids pandas openneuro-py

    echo "Environment installed. JAX version:"
    python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
else
    echo "Activating existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi
