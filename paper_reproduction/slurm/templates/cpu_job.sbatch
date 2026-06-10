#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source conf/narval.env
export JAX_PLATFORMS=cpu

mkdir -p logs

echo "=== $(date) === Job $SLURM_JOB_ID on $(hostname) ==="
echo "Script: $SCRIPT"
echo "Args:   ${ARGS:-}"

python -u "$SCRIPT" ${ARGS:-}

echo "=== $(date) === Done ==="
