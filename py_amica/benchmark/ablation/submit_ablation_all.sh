#!/bin/bash
# Submit the 4 ablation cells (2x2: num_models {1,3} x do_reject {0,1}) on ds004505,
# each a 25-subject array. Run ON fir (login-safe; only sbatch). The 4 arrays write to
# /scratch/sesma/amica_ablation/m{M}_reject{R}/ -> aggregate locally afterwards.
set -euo pipefail
cd "$(dirname "$0")"
for M in 1 3; do
  for R in 0 1; do
    sbatch --export=ALL,NUM_MODELS=${M},REJECT=${R} \
           --job-name="abl_m${M}_r${R}" \
           submit_ablation_ds004505.sh
  done
done
echo "==== submitted 4 cells; queue ===="
squeue --me -o "%.14i %.16j %.9T %.10M %R" | head -20
