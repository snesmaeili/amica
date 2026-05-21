#!/bin/bash
#SBATCH --job-name=amica_v3_numpy_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=1-25
#SBATCH --time=10:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/sesma/scratch/v3_numpy_cpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/v3_numpy_cpu_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-2000}"
export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-/scratch/$USER/amica_python_validation_v3}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004505 \
    --backend numpy \
    --device cpu \
    --n-iter "$AMICA_N_ITER" \
    --input-level bids \
    --schema-version v3
