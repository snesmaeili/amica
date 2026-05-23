#!/bin/bash
# FIR cluster environment setup for MNE-native synthetic AMICA benchmark.
# Clone of cc_benchmark/fir_env.sh with a synthetic-specific results dir.
# Ref: https://github.com/BabaSanfour/crash-course/tree/main/module_05_advanced_alliance_ai_workflows

# Caches to scratch
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export XDG_DATA_HOME="/scratch/$USER/.local/share"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$PIP_CACHE_DIR"

module purge
module load StdEnv/2023 || true
module load python/3.11
module load scipy-stack

# CUDA only needed for jax_gpu jobs but loading is harmless on CPU
module load cuda/12.6
module load cudnn

if [ -n "$CUDA_HOME" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi

# MNE sample dataset cache (downloaded once per scratch; ~50 MB)
export MNE_DATA="/scratch/$USER/mne_data"
mkdir -p "$MNE_DATA"

# Thread tuning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Synthetic-specific results dir (do NOT collide with the real-data dir)
export AMICA_SYNTH_RESULTS_DIR="${AMICA_SYNTH_RESULTS_DIR:-/scratch/$USER/amica_python_synthetic_v1}"
mkdir -p "$AMICA_SYNTH_RESULTS_DIR"

# Reuse the real-data venv -- amica-python is the same package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv_fir"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: expected venv at $VENV_PATH (set up by cc_benchmark/fir_env.sh on its first run)." >&2
    echo "Submit a CPU job that sources cc_benchmark/fir_env.sh first to provision it." >&2
    exit 1
fi
source "$VENV_PATH/bin/activate"
echo "[fir_env_synthetic] venv $VENV_PATH activated; AMICA_SYNTH_RESULTS_DIR=$AMICA_SYNTH_RESULTS_DIR"
