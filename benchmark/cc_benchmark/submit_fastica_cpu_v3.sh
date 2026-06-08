#!/bin/bash
#SBATCH --job-name=amica_v3_fastica
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=1-25
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/sesma/scratch/v3_fastica_%a_%j.out
#SBATCH --error=/home/sesma/scratch/v3_fastica_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-/scratch/$USER/amica_python_validation_v3}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"

# FastICA via MNE: tol=1e-6, fun='logcosh' (super-Gaussian leaning, fits EEG).
# Shared max_iter=5000 ceiling -- early-stops at tolerance.
python -m amica_python.benchmark.comparators \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --method fastica \
    --bids-root "$BIDS_ROOT_DS4505" \
    --output-dir "$AMICA_RESULTS_DIR" \
    --input-level bids \
    --n-components 64 \
    --random-state 42 \
    --max-iter 5000
