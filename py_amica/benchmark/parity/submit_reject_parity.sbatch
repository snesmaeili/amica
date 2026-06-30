#!/bin/bash
#SBATCH --job-name=amica17_reject_parity
#SBATCH --account=def-kjerbi
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/sesma/fortran_parity/reject_%j.out
#SBATCH --error=/scratch/sesma/fortran_parity/reject_%j.err
set -euo pipefail

# Single-model likelihood-rejection parity (Fortran AMICA 1.7 do_reject vs pyamica).
# Identical contaminated synth6 data (planted spikes) + Fortran's exact sphere/mean/fix_init
# + matched params -> compare Fortran's rejected set ({LLt total == 0}) and final W against
# pyamica's sample_mask_. Builds on the machine-precision no-rejection parity in
# FORTRAN_PARITY_PROVENANCE.md. NOT bit-identical: Fortran reads float32 data.fdt, the EM
# diverges at ~1e-4, and borderline-LL samples may flip — target is high set overlap +
# near-complete spike recall + matched |r|, not identity.

# REPO = a checkout of THIS branch (feat/reject-parity-harness: the rejection code +
# this harness). VENV = an existing .venv_fir (built on a login node with internet);
# we only re-link py_amica to REPO (--no-deps, no internet needed). CPU-only:
# the compare forces JAX_PLATFORM_NAME=cpu, so no CUDA modules are loaded.
REPO="${AMICA_REPO:-/scratch/sesma/amica-reject}"
VENV="${AMICA_VENV:-/scratch/sesma/pyamica/.venv_fir}"
BIN=/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17     # archived, validated 2026-06-09
WORK=/scratch/sesma/fortran_parity/reject_synth6

setup_py_env() {
    module purge
    module load StdEnv/2023 python/3.11 scipy-stack
    export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    source "$VENV/bin/activate"
    pip install -e "$REPO" --no-deps -q   # re-link py_amica to the rejection checkout
}

echo "############ PHASE 0: prep contaminated synth6 + do_reject param (pyamica env) ############"
setup_py_env
python -c "import sys, numpy, py_amica; print('py', sys.executable, '| py_amica', py_amica.__file__)"
mkdir -p "$WORK"
python -m py_amica.benchmark.parity.run_fortran_parity prep \
    --workdir "$WORK" --dataset synth6 --n-channels 6 --n-samples 20000 \
    --m 3 --do-newton 0 --max-iter 300 \
    --reject --spike-frac 0.01 --spike-amp 8 --rejsig 3.0 --numrej 1 --rejstart 3 --rejint 2 \
    --container-data "$WORK/data" --container-out "$WORK/out"

echo "############ PHASE 1: run archived amica17 with do_reject (gcc/openmpi env) ############"
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
export OMP_NUM_THREADS=1
file "$BIN"
cd "$WORK"
rm -rf "$WORK/out"; mkdir -p "$WORK/out"
mpirun -np 1 "$BIN" "$WORK/amica.param"
echo "----- Fortran out.txt (last 5 lines) -----"
tail -5 "$WORK/out/out.txt" || true
echo "----- LLt written? (needed for the rejected-set comparison) -----"
ls -la "$WORK/out/LLt" || echo "  LLt MISSING (write_LLt failed)"

echo "############ PHASE 2: pyamica compare — rejected sets + W (pyamica env) ############"
setup_py_env
cd "$WORK"
python -m py_amica.benchmark.parity.run_fortran_parity compare --workdir "$WORK"
cp -f "$WORK/parity.json" "$WORK/parity_reject_synth6.json" 2>/dev/null || true
echo "############ DONE ############"
