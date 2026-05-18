#!/bin/bash
# Submit all AMICA benchmarks on Fir.
# Run: bash submit_all.sh
#
# Order: smoke test first, then CPU baselines, then GPU.
# The smoke test runs on the MNE sample dataset to validate
# the environment before launching the expensive array jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== AMICA Benchmark Suite ==="
echo "Working directory: $(pwd)"
echo ""

# Step 1: Smoke test (MNE sample, fast)
echo "1/4 Submitting smoke test (MNE sample, NumPy CPU, 50 iters)..."
SMOKE_JOB=$(sbatch --parsable submit_smoke_mne.sh)
echo "  Smoke test job: $SMOKE_JOB"
echo ""

# Step 2: NumPy CPU array (depends on smoke test passing)
echo "2/4 Submitting NumPy CPU array (25 subjects)..."
NUMPY_JOB=$(sbatch --parsable --dependency=afterok:$SMOKE_JOB submit_numpy_cpu.sh)
echo "  NumPy CPU job array: $NUMPY_JOB"
echo ""

# Step 3: JAX CPU array
echo "3/4 Submitting JAX CPU array (25 subjects)..."
JAX_CPU_JOB=$(sbatch --parsable --dependency=afterok:$SMOKE_JOB submit_jax_cpu.sh)
echo "  JAX CPU job array: $JAX_CPU_JOB"
echo ""

# Step 4: JAX GPU array
echo "4/4 Submitting JAX GPU array (25 subjects, H100)..."
JAX_GPU_JOB=$(sbatch --parsable --dependency=afterok:$SMOKE_JOB submit_jax_gpu.sh)
echo "  JAX GPU job array: $JAX_GPU_JOB"
echo ""

echo "=== All jobs submitted ==="
echo "Smoke test must pass before array jobs start."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $SMOKE_JOB,$NUMPY_JOB,$JAX_CPU_JOB,$JAX_GPU_JOB --format=JobID,JobName,State,Elapsed"
echo ""
echo "After completion, aggregate results with:"
echo "  python aggregate_results.py"
