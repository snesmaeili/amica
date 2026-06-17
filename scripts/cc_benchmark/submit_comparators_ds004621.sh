#!/bin/bash
#SBATCH --job-name=amica_ds004621_comp
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=1-42
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/sesma/scratch/ds004621_comp_%a_%j.out
#SBATCH --error=/home/sesma/scratch/ds004621_comp_%a_%j.err
#
# Picard + FastICA + Infomax on ds004621 (same subjects + --n-components 64 as the AMICA
# job). All three methods per subject (--method all). 128-ch / long resting recording, so a
# 3 h ceiling; each method early-stops at its own tolerance. Writes alongside the AMICA JSONs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source fir_env.sh

export AMICA_RESULTS_DIR="${DS4621_RESULTS_DIR:-/scratch/$USER/amica_ds004621_v3}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"

python -m amica_python.benchmark.comparators \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004621 \
    --method all \
    --output-dir "$AMICA_RESULTS_DIR" \
    --n-components 64 \
    --random-state 42 \
    --max-iter 5000
