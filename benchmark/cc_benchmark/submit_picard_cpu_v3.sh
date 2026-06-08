#!/bin/bash
#SBATCH --job-name=amica_v3_picard
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=1-25
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/sesma/scratch/v3_picard_%a_%j.out
#SBATCH --error=/home/sesma/scratch/v3_picard_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-/scratch/$USER/amica_python_validation_v3}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"

# Picard via MNE: tol=1e-6 (tightened from sklearn default 1e-4), ortho=False,
# extended=True. Shared max_iter=5000 ceiling -- early-stops at tolerance.
python -m amica_python.benchmark.comparators \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --method picard \
    --bids-root "$BIDS_ROOT_DS4505" \
    --output-dir "$AMICA_RESULTS_DIR" \
    --input-level bids \
    --n-components 64 \
    --random-state 42 \
    --max-iter 5000
