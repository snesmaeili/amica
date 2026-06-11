#!/bin/bash
#SBATCH --job-name=amica_v3_infomax
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --array=1-25
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-$AMICA_RESULTS_DIR}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"

# Infomax via MNE: extended=True, w_change=1e-7 (matches Picard tightness).
# Shared max_iter=5000 ceiling. Sub-01 hit cap at 5000; some subjects may
# converge earlier. If many hit the cap, lower w_change or raise max_iter.
python -m amica_python.benchmark.comparators \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --method infomax \
    --bids-root "$BIDS_ROOT_DS4505" \
    --output-dir "$AMICA_RESULTS_DIR" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components 64 \
    --random-state 42 \
    --max-iter 5000
