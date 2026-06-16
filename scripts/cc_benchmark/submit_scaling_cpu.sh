#!/bin/bash
# Tier-2 scaling sweep (CPU): peak host RSS vs n_samples (T) for the full-batch E-step (expected
# ~O(T)) and vs chunk_size at full T (expected bounded ~O(B), confirming the O(B*M*C*K) claim).
# AMICA-JAX on CPU, ds004505 sub-01, 60 iter (past the newt_start=50 Newton allocation).
#SBATCH --job-name=amica_scaling_cpu
#SBATCH --account=def-kjerbi
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/sesma/amica_scaling_logs/%x-%j.out
#SBATCH --error=/scratch/sesma/amica_scaling_logs/%x-%j.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO/scripts/cc_benchmark"
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"
export AMICA_RESULTS_DIR=/scratch/sesma/amica_scaling/cpu
mkdir -p "$AMICA_RESULTS_DIR"

run() {
  local tag="$1"; shift
  echo "--- $tag : $* ---"
  python run_one_subject.py --dataset ds004505 --subject 1 --input-level bids \
    --backend jax --device cpu --schema-version v3 --n-iter 60 \
    --output-dir "$AMICA_RESULTS_DIR/$tag" "$@"
}

echo "=== RSS vs T (C=64, full-batch / chunk=None) ==="
for D in 60 150 300 600; do run "Tsec-${D}_fullbatch" --n-components 64 --duration-sec "$D"; done
run "Tfull_fullbatch" --n-components 64

echo "=== RSS vs chunk_size (full T, C=64) ==="
for CH in 1024 4096 16384 65536; do run "chunk-${CH}_Tfull" --n-components 64 --chunk-size "$CH"; done
# chunk=None (full-batch) at full T == Tfull_fullbatch above

echo "=== DONE ==="; ls -R "$AMICA_RESULTS_DIR" | head -40
