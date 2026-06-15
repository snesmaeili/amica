#!/bin/bash
# GPU VRAM head-to-head: AMICA-JAX (chunk_size=auto = the paper config) vs pyamica vs scott
# on ONE dedicated H100. Each framework's ALLOCATED peak (XLA peak_bytes_in_use /
# torch.cuda.max_memory_allocated) with preallocation/caching disabled, plus the neutral NVML
# whole-GPU peak (--nvml-crosscheck). 1 subject, low iter (memory is iteration-independent).
# Produces per-impl result JSONs only; aggregate + figure run LOCALLY after rsync.
#
# PREREQUISITE (gate-checked before submitting): .venv_competitors must have a CUDA-enabled torch.
# If torch.cuda.is_available() is False there, pyamica/scott can't run on GPU.
#SBATCH --job-name=amica_mem_gpu
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
source "$REPO/scripts/cc_benchmark/fir_env.sh"   # python venv + cuda/cudnn + BIDS_ROOT + caches

echo "=== env ==="
echo "python (amica venv): $(which python)"; python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1
echo "competitors torch:"
"$REPO/.venv_competitors/bin/python" -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1

echo "=== run comparator on GPU: amica-jax(auto) + pyamica + scott ==="
python scripts/comparator/implementation_perf.py \
    --dataset ds004505 --subject 1 --input-level bids \
    --n-components 64 --max-iter 100 \
    --amica-device gpu --competitor-device gpu --nvml-crosscheck \
    --amica-chunk-size auto \
    --skip amica_python_jax amica_python_numpy neuromechanist_numpy fortran_amica17 \
    --out-tag ds004505_sub-01_mem_gpu

echo "=== result JSONs ==="
ls -l "$REPO/results/comparator/ds004505_sub-01_mem_gpu/" 2>&1
echo "=== DONE ==="
