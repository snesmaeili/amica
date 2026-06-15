#!/bin/bash
# Build .venv_competitors with the 3 external AMICA reimplementations used by the
# cross-implementation MEMORY comparison (benchmark/comparator). Run ONCE per site.
#
#   pyamica  (DerAndereJohannes)  PyTorch     -> import pyamica
#   amica    (scott-huberty)      PyTorch     -> import amica
#   pyAMICA  (neuromechanist)     pure NumPy  -> import pyAMICA
#
# git+pip needs internet, so run this on the LOGIN node — it is a one-time ENV BUILD
# (not compute), which is the supported use of pip on a login node. On Alliance the
# `--no-index torch` wheel is CUDA-capable and serves both the CPU and GPU comparisons.
#
#   bash setup_competitors.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/cc_benchmark/
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${COMPETITORS_VENV_DIR:-$REPO_ROOT/.venv_competitors}"

# Caches off $HOME (Alliance quota), mirroring fir_env.sh.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    virtualenv --no-download "$VENV" 2>/dev/null || python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# torch: Alliance wheelhouse wheel (--no-index) is CUDA-enabled and works for BOTH the CPU
# and GPU comparisons; fall back to PyPI off-Alliance.
echo "Installing torch ..."
pip install --no-index torch 2>/dev/null || pip install torch

# The 3 competitor AMICA implementations (all pip-installable from GitHub):
echo "Installing the 3 competitor implementations ..."
pip install "git+https://github.com/DerAndereJohannes/pyamica.git"
pip install "git+https://github.com/scott-huberty/amica-python.git"
pip install "git+https://github.com/neuromechanist/pyAMICA.git"

# Optional: NVML neutral cross-check for the GPU comparison (enable with AMICA_MEM_NVML=1).
pip install nvidia-ml-py 2>/dev/null || echo "(nvidia-ml-py not installed; NVML cross-check stays off)"

echo "=== verify imports ==="
python -c "import torch, pyamica, amica, pyAMICA; print('OK — torch', torch.__version__, '| pyamica + amica(scott) + pyAMICA(neuromechanist) all import')"
echo "competitors venv ready: $VENV"
