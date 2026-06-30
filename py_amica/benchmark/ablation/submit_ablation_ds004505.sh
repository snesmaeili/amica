#!/bin/bash
#SBATCH --job-name=amica_ablation
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-25
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/sesma/amica_ablation/%x_%A_%a.out
#SBATCH --error=/scratch/sesma/amica_ablation/%x_%A_%a.err
set -euo pipefail

# Real-EEG impact ablation: AMICA on ds004505, ONE of the 2x2 cells
# {num_models in 1,3} x {do_reject 0,1}. Parameterized by env vars NUM_MODELS + REJECT,
# set on the sbatch command line (see submit_ablation_all.sh). Per-subject v3 JSON
# (MIR / dipolarity / PMI / ICLabel / LL / n_rejected) -> aggregate locally to the table.
#
# Self-contained: reuses an existing .venv_fir (built on a login node) and re-links
# py_amica to this checkout (the M>1-rejection branch) with --no-deps (no internet).

REPO="${AMICA_REPO:-/scratch/sesma/amica-reject}"
VENV="${AMICA_VENV:-/scratch/sesma/pyamica/.venv_fir}"
NUM_MODELS="${NUM_MODELS:-1}"
REJECT="${REJECT:-0}"
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
export AMICA_COMPUTE_DIPOLES=1   # enable the dipolarity metric (nd_5/nd_10 percent)

CELL="m${NUM_MODELS}_reject${REJECT}"
OUT="/scratch/sesma/amica_ablation/${CELL}"
mkdir -p "$OUT"

module purge
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.6 cudnn
[ -n "${CUDA_HOME:-}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "$VENV/bin/activate"
# Use the M>1-rejection checkout's py_amica via PYTHONPATH (NOT a per-task
# `pip install -e` — that races on the shared venv's editable-finder file under array
# concurrency). PYTHONPATH is prepended to sys.path so it shadows the venv's py_amica.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

REJ_FLAG=""
[ "$REJECT" = "1" ] && REJ_FLAG="--do-reject --rejsig 3.0 --rejstart 2 --rejint 3 --numrej 5"

echo "==== ablation cell ${CELL} subject ${SLURM_ARRAY_TASK_ID} ===="
python -c "import sys, py_amica; print('py_amica', py_amica.__file__)"
python -m py_amica.benchmark.runner \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" \
    --backend jax --device gpu --schema-version v3 \
    --n-components 64 --n-iter 3000 \
    --num-models "$NUM_MODELS" $REJ_FLAG \
    --output-dir "$OUT"
echo "==== done ${CELL} sub-${SLURM_ARRAY_TASK_ID} ===="
