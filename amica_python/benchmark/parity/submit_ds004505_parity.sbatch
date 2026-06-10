#!/bin/bash
#SBATCH --job-name=amica17_ds4505_parity
#SBATCH --account=def-kjerbi
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/sesma/fortran_parity/ds4505_sub01_%j.out
#SBATCH --error=/scratch/sesma/fortran_parity/ds4505_sub01_%j.err
set -euo pipefail

REPO=/scratch/sesma/amica-python
BIN=/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17     # archived, validated
WORK=/scratch/sesma/fortran_parity/ds004505_sub01
ENVSH="$REPO/benchmark/cc_benchmark/fir_env.sh"

echo "############ PHASE 0: prep ds004505 sub-01 → PCA-64 (amica-python env) ############"
module purge
source "$ENVSH"
python -c "import sys, mne, numpy; print('py', sys.executable, 'mne', mne.__version__)"
mkdir -p "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity prep \
    --workdir "$WORK" --dataset ds004505 --subject 1 --k-pca 64 \
    --m 3 --do-newton 1 --max-iter 100 \
    --container-data "$WORK/data" --container-out "$WORK/out"

echo "############ PHASE 1: run archived amica17 (gcc/openmpi env) ############"
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
export OMP_NUM_THREADS=1
file "$BIN"
cd "$WORK"
rm -rf "$WORK/out"
mpirun -np 1 "$BIN" "$WORK/amica.param"
echo "----- Fortran out.txt (last 3 lines) -----"
tail -3 "$WORK/out/out.txt" || true

echo "############ PHASE 2: amica-python compare (amica-python env) ############"
module purge
source "$ENVSH"
cd "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity compare --workdir "$WORK"
cp -f "$WORK/parity.json" "$WORK/parity_ds004505_sub01.json" 2>/dev/null || true
echo "############ DONE ############"
