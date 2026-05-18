#!/bin/bash
# FIR cluster environment setup for AMICA benchmarking
# Ref: https://github.com/BabaSanfour/crash-course/tree/main/module_05_advanced_alliance_ai_workflows

# ── Module loads ──
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
RESULTS_DIR="/project/rrg-kjerbi/projects/amica_python_validation/outputs"
export AMICA_RESULTS_DIR="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ── Virtual environment ──
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

# Get the repo root (two levels up from scripts/cc_benchmark/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
