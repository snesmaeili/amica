#!/bin/bash
# Cross-implementation MEMORY comparison — GPU peak VRAM (portable capsule version).
#
# AMICA-JAX (chunk_size=auto) vs pyamica vs scott-huberty on ONE GPU. Reports each framework's
# ALLOCATED peak (XLA peak_bytes_in_use / torch.cuda.max_memory_allocated) with preallocation
# and caching disabled, so the numbers are true demand, not the reserved pool. 1 subject, low
# iter (memory is iteration-independent). Aggregate + figure run locally afterwards (see README).
#
# PREREQUISITE: a CUDA-enabled torch in .venv_competitors. On Alliance the module wheel
# (pip install --no-index torch, what setup_competitors.sh uses) is already CUDA-capable.
# The job prints torch.cuda.is_available(); if False, only AMICA-JAX's VRAM is recorded.
#
# The #SBATCH lines are fir defaults; override per-site via submit_all.sh (--account/--partition/--gres).
#SBATCH --job-name=amica_mem_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh              # modules (incl. cuda/cudnn) + .venv_fir + env.local
REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"

echo "=== env ==="; echo "amica python: $(which python)"; python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1
"$REPO_ROOT/.venv_competitors/bin/python" -c "import torch; print('competitors torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1

# GPU results in their own subdir (CPU job uses .../comparator/cpu).
export AMICA_COMPARATOR_RESULTS="${AMICA_COMPARATOR_RESULTS:-${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/comparator/gpu}"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

# Optional NVML whole-GPU cross-check (needs pynvml in BOTH venvs); off by default.
NVML_OPT=""
[ "${AMICA_MEM_NVML:-0}" = "1" ] && NVML_OPT="--nvml-crosscheck"

# Optional custom run tag; default is the orchestrator's '<dataset>_sub-NN'.
TAG_OPT=""; [ -n "${AMICA_MEM_GPU_TAG:-}" ] && TAG_OPT="--out-tag ${AMICA_MEM_GPU_TAG}"

echo "=== GPU memory comparison: dataset=${AMICA_MEM_DATASET:-ds004505} subject=${AMICA_MEM_SUBJECT:-1} ==="
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter "${AMICA_MEM_ITER:-100}" \
    --amica-device gpu --competitor-device gpu $NVML_OPT \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    $TAG_OPT \
    --skip amica_python_jax amica_python_numpy neuromechanist_numpy fortran_amica17

echo "=== DONE. Results under $AMICA_COMPARATOR_RESULTS/ . Aggregate locally: ==="
echo "  python ../comparator/aggregate_pilot.py --root '$AMICA_COMPARATOR_RESULTS' --impls all"
