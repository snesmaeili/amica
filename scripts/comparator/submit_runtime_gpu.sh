#!/bin/bash
# Fair GPU per-iteration runtime via the 2-POINT method, for the cross-implementation runtime figure
# (companion to fig_mem_comparison / fig_impl_runtime). Runs the comparator on one H100 for the three
# implementations that have a GPU path -- amica-python JAX (chunk_size=auto) + scott-huberty (PyTorch)
# + pyamica (PyTorch) -- at TWO iteration counts (100 and 600). Steady-state per-iter is then
# (T_600 - T_100) / (600 - 100), which cancels JAX's one-time XLA JIT compile (a raw 100-iter GPU fit
# is ~90% compile for JAX and would unfairly understate it). Same node -> clean 2-point.
# Produces per-impl result JSONs only; the 2-point + figure run LOCALLY after rsync.
#SBATCH --job-name=amica_runtime_gpu_2pt
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/sesma/amica_mem_logs/%x-%j.out
#SBATCH --error=/scratch/sesma/amica_mem_logs/%x-%j.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO"
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"

echo "=== env ==="; python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1
"$REPO/.venv_competitors/bin/python" -c "import torch; print('competitors torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1

for IT in 100 600; do
  echo "=== comparator GPU @ ${IT} iter ==="
  python scripts/comparator/implementation_perf.py \
      --dataset ds004505 --subject 1 --input-level bids \
      --n-components 64 --max-iter ${IT} \
      --amica-device gpu --competitor-device gpu --amica-chunk-size auto \
      --skip amica_python_jax amica_python_numpy neuromechanist_numpy fortran_amica17 \
      --out-tag ds004505_sub-01_rt_gpu_${IT}
done

echo "=== result JSONs ==="
ls -l "$REPO/results/comparator/ds004505_sub-01_rt_gpu_100/" "$REPO/results/comparator/ds004505_sub-01_rt_gpu_600/" 2>&1
echo "=== DONE ==="
