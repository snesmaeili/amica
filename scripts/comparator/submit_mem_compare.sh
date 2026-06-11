#!/bin/bash
# Cross-implementation MEMORY comparison on ds004505 sub-01 (1 subject, low iter).
# Memory (peak RSS, peak VRAM) is set by data dimensions + algorithm buffers, NOT
# iteration count, so ~100 iter captures the peak (incl. the newt_start=50 Newton
# buffers) at a tiny fraction of the v3 cost. Produces per-impl result JSONs only;
# aggregate + figure are run LOCALLY after rsync (the cluster is compute-only).
#SBATCH --job-name=amica_mem_compare
#SBATCH --account=def-kjerbi
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/sesma/amica_mem_logs/%x-%j.out
#SBATCH --error=/scratch/sesma/amica_mem_logs/%x-%j.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO"

# Python env (venv + BIDS_ROOT_DS4505=raw_bids + scratch caches), then the Fortran
# run modules (openmpi for mpirun, flexiblas for amica17's BLAS) loaded on top.
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"
module load openmpi/4.1.5 flexiblas

export AMICA17_BIN=/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17
export GNU_TIME_BIN=/usr/bin/time

echo "=== env ==="
echo "python : $(which python)"; python --version
echo "mpirun : $(which mpirun)"
echo "amica17: $AMICA17_BIN"; ls -l "$AMICA17_BIN"
echo "BIDS   : ${BIDS_ROOT_DS4505:-unset}"
echo "threads: OMP=${OMP_NUM_THREADS:-unset} (run_fortran forces OMP=1 for amica17)"

echo "=== run comparator: 5 impls (amica-jax CPU, pyamica, scott, neuromechanist, fortran) ==="
python scripts/comparator/implementation_perf.py \
    --dataset ds004505 --subject 1 --input-level bids \
    --n-components 64 --max-iter 100 \
    --amica-device cpu --competitor-device cpu \
    --include-fortran \
    --skip amica_python_numpy \
    --out-tag ds004505_sub-01_mem

echo "=== result JSONs ==="
ls -l "$REPO/results/comparator/ds004505_sub-01_mem/" 2>&1
echo "=== DONE ==="
