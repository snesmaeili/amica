#!/bin/bash
#SBATCH --job-name=amica17_mklfix
#SBATCH --account=def-kjerbi
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/sesma/fortran_parity/mklfix50k_%j.out
#SBATCH --error=/scratch/sesma/fortran_parity/mklfix50k_%j.err
set -euo pipefail

REPO=/scratch/sesma/amica-python
BUILD=/scratch/sesma/fortran_parity/amica17_mkl_build
WORK=/scratch/sesma/fortran_parity/mklfix50k
ENVSH="$REPO/benchmark/cc_benchmark/fir_env.sh"

echo "############ PHASE 0: prep (numpy / amica-python env) ############"
module purge
source "$ENVSH"
python -c "import sys, numpy; print('py', sys.executable, 'numpy', numpy.__version__)"
mkdir -p "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity prep \
    --workdir "$WORK" --dataset synth6 --n-channels 6 --n-samples 50000 \
    --m 3 --do-newton 1 --max-iter 1000 --seed 0 \
    --container-data "$WORK/data" --container-out "$WORK/out"

echo "############ PHASE 1: build amica17 (-DMKL, patched) + run (gcc/openmpi env) ############"
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
export OMP_NUM_THREADS=1
cd "$BUILD"
bash build.sh
file "$BUILD/amica17"
cd "$WORK"
rm -rf "$WORK/out"
mpirun -np 1 "$BUILD/amica17" "$WORK/amica.param"
echo "----- Fortran out.txt (last 3 lines) -----"
tail -3 "$WORK/out/out.txt" || true

echo "############ PHASE 2: compare to amica-python (amica-python env) ############"
module purge
source "$ENVSH"
cd "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity compare --workdir "$WORK"
echo "############ DONE ############"
