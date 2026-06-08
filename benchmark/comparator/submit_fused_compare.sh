#!/bin/bash
# Stage 3E — fused-AMICA (CPU+GPU) vs pyamica-torch (CPU) re-comparison on fir.
#
# Re-runs the comparator head-to-head with the Stage 3D fused E-step
# (branch jax-performance-pass, estep="auto" -> fused by default), to measure
# whether the fused path closes/beats pyamica-torch on CPU and to capture the
# GPU advantage. Subjects {1,2,4}, 10-min crop / 500 iter (matches the pilot so
# numbers are directly comparable to comparator_pilot_cluster/).
#
# Each job sources scripts/cc_benchmark/fir_env.sh FIRST. That is the VALIDATED
# v3 module recipe: module purge; StdEnv/2023; python/3.11; scipy-stack;
# cuda/12.6; cudnn; sets XLA cuda_data_dir; and activates .venv_fir. This is the
# piece the earlier comparator pilot submit script omitted, and it guarantees the
# GPU job has CUDA/cuDNN exactly as the paper runs did.
#
# Three series, distinct --out-tag dirs so the amica CPU and GPU series (both
# write amica_python_jax_*.json) do not collide:
#   fused_amica_cpu    def-kjerbi_cpu  (auto part.)        JAX cpu    only amica_python_jax
#   fused_amica_gpu    def-kjerbi_gpu  gpubase_bygpu_b1    JAX cuda   only amica_python_jax
#   fused_pyamica_cpu  def-kjerbi_cpu  (auto part.)        torch cpu  only pyamica_torch
#
# Run ON fir (login-safe; only calls sbatch):
#   ssh fir 'cd /scratch/sesma/amica-python && bash scripts/comparator/submit_fused_compare.sh'
set -euo pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO"
mkdir -p logs

VF="$REPO/.venv_fir/bin/python"
VC="$REPO/.venv_competitors/bin/python"
BIDS=/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids
ARRAY=1,2,4
COMMON="--dataset ds004505 --n-components 64 --seeds 0 --resample-sfreq 250 --max-iter 500 --duration-sec 600"

# ---------- 1) AMICA fused, CPU ----------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fused_amica_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=$ARRAY
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$REPO"
source scripts/cc_benchmark/fir_env.sh
SID=\$SLURM_ARRAY_TASK_ID
TAG=cmp_fused/amica_cpu/ds004505_sub-\$(printf '%02d' \$SID)
JAX_PLATFORMS=cpu BIDS_ROOT_DS4505=$BIDS AMICA_PYTHON_VENV=$VF COMPETITORS_VENV=$VC \\
  $VF -u scripts/comparator/implementation_perf.py \\
    $COMMON --subject \$SID --amica-device cpu --out-tag \$TAG \\
    --skip amica_python_numpy pyamica_torch scott_huberty_torch neuromechanist_numpy
EOF

# ---------- 2) AMICA fused, GPU (H100) ----------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fused_amica_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=$ARRAY
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$REPO"
source scripts/cc_benchmark/fir_env.sh
SID=\$SLURM_ARRAY_TASK_ID
TAG=cmp_fused/amica_gpu/ds004505_sub-\$(printf '%02d' \$SID)
JAX_PLATFORMS=cuda BIDS_ROOT_DS4505=$BIDS AMICA_PYTHON_VENV=$VF COMPETITORS_VENV=$VC \\
  $VF -u scripts/comparator/implementation_perf.py \\
    $COMMON --subject \$SID --amica-device gpu --out-tag \$TAG \\
    --skip amica_python_numpy pyamica_torch scott_huberty_torch neuromechanist_numpy
EOF

# ---------- 3) pyamica-torch, CPU (fresh; controls node variance) ----------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fused_pyamica_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=$ARRAY
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$REPO"
source scripts/cc_benchmark/fir_env.sh
SID=\$SLURM_ARRAY_TASK_ID
TAG=cmp_fused/pyamica_cpu/ds004505_sub-\$(printf '%02d' \$SID)
JAX_PLATFORMS=cpu BIDS_ROOT_DS4505=$BIDS AMICA_PYTHON_VENV=$VF COMPETITORS_VENV=$VC \\
  $VF -u scripts/comparator/implementation_perf.py \\
    $COMMON --subject \$SID --amica-device cpu --out-tag \$TAG \\
    --skip amica_python_jax amica_python_numpy scott_huberty_torch neuromechanist_numpy
EOF

echo "===== submitted; current queue ====="
squeue --me -o "%.12i %.22j %.16a %.9T %.11M %.11l %.6D %R"
