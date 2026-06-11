#!/bin/bash
# One-command driver for the AMICA-Python Compute Canada benchmark
# (6 ICA methods x 25 subjects on ds004505 -> the zenodo-paper results).
#
# Usage, on an Alliance/Compute-Canada login node:
#   cd benchmark/cc_benchmark
#   cp env.template env.local      # then edit env.local: your allocation, GPU, data path
#   bash submit_all.sh
#
# It (1) sources env.local, (2) runs the MNE smoke gate, (3) submits the six
# canonical 25-subject arrays, passing your --account/--partition/--gres on the
# sbatch command line (CLI overrides the fir defaults baked into each script).
# By default the arrays are gated on the smoke job (--dependency=afterok); set
# AMICA_GATE_ON_SMOKE=0 in env.local to submit everything immediately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ -f env.local ] || { echo "ERROR: copy env.template -> env.local and edit it first."; exit 1; }
# shellcheck disable=SC1091
source env.local
: "${AMICA_ACCOUNT:?set AMICA_ACCOUNT in env.local}"
: "${AMICA_GPU_ACCOUNT:?set AMICA_GPU_ACCOUNT in env.local}"
: "${BIDS_ROOT_DS4505:?set BIDS_ROOT_DS4505 in env.local}"

# Per-site sbatch overrides (CLI beats the #SBATCH fir defaults in each script).
GPU_OPTS=(--account="$AMICA_GPU_ACCOUNT")
[ -n "${AMICA_GPU_PARTITION:-}" ] && GPU_OPTS+=(--partition="$AMICA_GPU_PARTITION")
[ -n "${AMICA_GPU_GRES:-}" ]      && GPU_OPTS+=(--gres="$AMICA_GPU_GRES")
CPU_OPTS=(--account="$AMICA_ACCOUNT")
[ -n "${AMICA_CPU_PARTITION:-}" ] && CPU_OPTS+=(--partition="$AMICA_CPU_PARTITION")

# (1) smoke gate -- capture its job id for the afterok dependency.
smoke=$(sbatch "${CPU_OPTS[@]}" --parsable submit_smoke_mne.sh)
echo "smoke job: $smoke"
DEP=()
[ "${AMICA_GATE_ON_SMOKE:-1}" = "1" ] && DEP=(--dependency=afterok:"$smoke")

# (2) the six canonical 25-subject arrays.
sbatch "${GPU_OPTS[@]}" "${DEP[@]}" submit_jax_gpu_v3.sh
sbatch "${CPU_OPTS[@]}" "${DEP[@]}" submit_jax_cpu_v3.sh
sbatch "${CPU_OPTS[@]}" "${DEP[@]}" submit_numpy_cpu_v3.sh
sbatch "${CPU_OPTS[@]}" "${DEP[@]}" submit_picard_cpu_v3.sh
sbatch "${CPU_OPTS[@]}" "${DEP[@]}" submit_infomax_cpu_v3.sh
sbatch "${CPU_OPTS[@]}" "${DEP[@]}" submit_fastica_cpu_v3.sh

cat <<EOF

Submitted. Watch the queue with:  squeue --me
When ALL arrays finish, aggregate the per-subject JSONs into the 3 paper CSVs:

  python -m amica_python.benchmark.aggregate \\
      --results-dir "${AMICA_RESULTS_DIR:-/scratch/\$USER/amica_python_validation_v3}" \\
      --output-dir  "${AMICA_RESULTS_DIR:-/scratch/\$USER/amica_python_validation_v3}"

then render the paper figures from those CSVs (see REPRODUCING.md Step 7).
EOF
