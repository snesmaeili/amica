#!/bin/bash
#SBATCH --job-name=amica_ds004504_comp
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=37-65
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Picard + FastICA + Infomax on ds004504 (same subjects + --n-components 15 as the AMICA
# job). All three methods per subject (--method all; cheap at 19 channels). Shared 5000-
# iteration ceiling; each early-stops at its own tolerance. Identical per-site preprocessing
# to AMICA (shared runner.load_data/preprocess). Writes alongside the AMICA JSONs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_RESULTS_DIR="${DS4504_RESULTS_DIR:-/scratch/$USER/amica_ds004504_v3}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"

python -m amica_python.benchmark.comparators \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004504 \
    --method all \
    --output-dir "$AMICA_RESULTS_DIR" \
    --n-components 15 \
    --random-state 42 \
    --max-iter 5000
