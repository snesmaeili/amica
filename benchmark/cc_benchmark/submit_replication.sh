#!/bin/bash
# Driver for the 3-dataset MIR/dipolarity REPLICATION (ds004504 + ds004621), the second
# and third real datasets that accompany the ds004505 headline (submit_all.sh).
#
# Each dataset runs AMICA-JAX (GPU) + the 3 MNE comparators (CPU) on the SAME per-site
# preprocessed input: 1 Hz high-pass, 1-100 Hz band-pass, local-mains notch (50 Hz for the
# European ds004504/ds004621), and a resample to 250 Hz -- all handled inside
# runner.load_data/preprocess, so AMICA and the comparators are byte-identical. ds004505
# (the headline) is produced by submit_all.sh and is NOT re-run here.
#
# Usage on an Alliance/Compute-Canada login node:
#   cd benchmark/cc_benchmark
#   cp env.template env.local      # edit allocation/GPU + BIDS_ROOT_DS4504 + BIDS_ROOT_DS4621
#   bash download_ds004504.sh && bash download_ds004621.sh
#   bash submit_replication.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ -f env.local ] || { echo "ERROR: copy env.template -> env.local and edit it first."; exit 1; }
# shellcheck disable=SC1091
source env.local
: "${AMICA_ACCOUNT:?set AMICA_ACCOUNT in env.local}"
: "${AMICA_GPU_ACCOUNT:?set AMICA_GPU_ACCOUNT in env.local}"
: "${BIDS_ROOT_DS4504:?set BIDS_ROOT_DS4504 in env.local}"
: "${BIDS_ROOT_DS4621:?set BIDS_ROOT_DS4621 in env.local}"

# Per-site sbatch overrides (CLI beats the #SBATCH fir defaults in each script).
GPU_OPTS=(--account="$AMICA_GPU_ACCOUNT")
[ -n "${AMICA_GPU_PARTITION:-}" ] && GPU_OPTS+=(--partition="$AMICA_GPU_PARTITION")
[ -n "${AMICA_GPU_GRES:-}" ]      && GPU_OPTS+=(--gres="$AMICA_GPU_GRES")
CPU_OPTS=(--account="$AMICA_ACCOUNT")
[ -n "${AMICA_CPU_PARTITION:-}" ] && CPU_OPTS+=(--partition="$AMICA_CPU_PARTITION")

# ds004504 (19-ch resting) -- AMICA-GPU + 3 comparators, n_components 15.
sbatch "${GPU_OPTS[@]}" submit_jax_gpu_ds004504.sh
sbatch "${CPU_OPTS[@]}" submit_comparators_ds004504.sh
# ds004621 (128-ch resting) -- AMICA-GPU + 3 comparators, n_components 64.
sbatch "${GPU_OPTS[@]}" submit_jax_gpu_ds004621.sh
sbatch "${CPU_OPTS[@]}" submit_comparators_ds004621.sh

cat <<EOF

Submitted the 4 replication arrays (ds004504 + ds004621; AMICA-GPU + comparators).
Watch:  squeue --me
When ALL arrays finish, aggregate each dataset's per-subject JSONs into a CSV:

  python -m amica_python.benchmark.aggregate \\
      --results-dir "\${DS4504_RESULTS_DIR:-/scratch/\$USER/amica_ds004504_v3}" \\
      --output-dir  "\${DS4504_RESULTS_DIR:-/scratch/\$USER/amica_ds004504_v3}"
  python -m amica_python.benchmark.aggregate \\
      --results-dir "\${DS4621_RESULTS_DIR:-/scratch/\$USER/amica_ds004621_v3}" \\
      --output-dir  "\${DS4621_RESULTS_DIR:-/scratch/\$USER/amica_ds004621_v3}"

then render the replication figures from those CSVs (see REPRODUCING.md).
EOF
