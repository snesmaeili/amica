#!/bin/bash
# Submit all benchmarks

echo "Submitting NumPy CPU jobs..."
sbatch submit_numpy_cpu.sh

echo "Submitting JAX CPU jobs..."
sbatch submit_jax_cpu.sh

echo "Submitting JAX GPU jobs..."
sbatch submit_jax_gpu.sh
