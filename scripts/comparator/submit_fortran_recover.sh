#!/bin/bash
# One-off: re-run ONLY Fortran amica17 on the saved comparator input (no re-preprocess),
# overwriting its result JSON. Use after a run_fortran param/runtime fix.
#SBATCH --job-name=amica_fortran_recover
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/sesma/amica_mem_logs/%x-%j.out
#SBATCH --error=/scratch/sesma/amica_mem_logs/%x-%j.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO"
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"   # python venv (numpy/psutil) + caches
module load openmpi/4.1.5 flexiblas              # amica17 runtime (mpirun + BLAS)
export AMICA17_BIN=/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17
export GNU_TIME_BIN=/usr/bin/time

D="$REPO/results/comparator/ds004505_sub-01_mem"
python scripts/comparator/runners/run_fortran.py \
    --input "$D/input_sub-01.npz" \
    --output "$D/fortran_amica17_sub-01_seed0_result.json" \
    --config '{"max_iter":100,"n_mix":3,"lrate":0.1,"do_newton":true,"seed":0}'

echo "=== fortran result ==="
grep -E '"(implementation|peak_rss_gb|delta_rss_gb|n_iter|fit_time_s|error)"' \
    "$D/fortran_amica17_sub-01_seed0_result.json" 2>&1
echo "=== DONE ==="
