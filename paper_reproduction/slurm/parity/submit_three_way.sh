#!/bin/bash
#SBATCH --job-name=3way-parity
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/3way-parity-%j.out
#SBATCH --error=logs/3way-parity-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load modules for Fortran amica17_narval
module load gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1

# Python environment
source conf/narval.env
export JAX_PLATFORMS=cpu

# Fortran runtime
ulimit -s unlimited
export OMP_STACKSIZE=512M

mkdir -p logs results/parity

DATASET="${1:-mne}"
L5_ITERS="${2:-200}"

echo "=== $(date) === Job $SLURM_JOB_ID on $(hostname) ==="
echo "Dataset: $DATASET, L5 iters: $L5_ITERS"
echo "Fortran binary: /home/sesma/refs/sccn-amica/amica17_narval"

python -u scripts/parity/three_way_parity.py \
    --dataset "$DATASET" \
    --levels 1,2,3,4,5 \
    --l5-iters "$L5_ITERS" \
    --n-components 30

echo "=== $(date) === Done ==="
