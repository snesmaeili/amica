#!/bin/bash
# S6 — multi-model AMICA real-EEG demo on ds004505 (fir).
#
# Fits AMICA with H in {1,2,3} on subjects {1,4} and saves the model-probability
# time-course p(h|t) + LL + task events (run_multimodel_demo.py), so the local
# figure shows whether the active model tracks the task structure (Hsu 2018) and
# whether H>=2 beats H=1.
#
# GPU, def-kjerbi_gpu. Submitted FROM scripts/cc_benchmark so the v3 scripts'
# relative `source fir_env.sh` resolves (StdEnv/2023 + python/3.11 + scipy-stack
# + cuda/12.6 + cudnn + .venv_fir). Branch multimodel-amica must be checked out.
#
# Run ON fir (login-safe; only calls sbatch):
#   ssh fir 'cd /scratch/sesma/amica-python/scripts/cc_benchmark && bash submit_multimodel_demo.sh'
set -euo pipefail

CCDIR=/scratch/sesma/amica-python/scripts/cc_benchmark
OUT=/scratch/sesma/multimodel_demo
ARRAY=1,4
mkdir -p "$OUT" "$CCDIR/logs"

for H in 1 2 3; do
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mm_demo_M${H}
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
cd $CCDIR
source fir_env.sh
SID=\$SLURM_ARRAY_TASK_ID
JAX_PLATFORMS=cuda python run_multimodel_demo.py \\
    --subject \$SID --num-models ${H} --n-iter 2000 --n-components 64 \\
    --num-mix 3 --duration-sec 600 --resample 250 --output-dir $OUT
EOF
done

echo "===== submitted; current queue ====="
squeue --me -o "%.14i %.16j %.16a %.9T %.6D %R"
