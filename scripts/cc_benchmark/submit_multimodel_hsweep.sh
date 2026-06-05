#!/bin/bash
# Multi-model stationarity benchmark — ds004505 H-sweep on fir (GPU).
#
# Fits AMICA with H=1..10 on all 25 ds004505 subjects (N=16 components, FULL
# recording so the ~25*H*N^2 data-length rule supports H up to 10), saving the
# per-model artifacts the local metric driver needs. A 1-task smoke (sub-01,H=2)
# gates the full array via afterok.
#
# Array mapping (250 tasks): task t -> subject=((t-1)/10)+1, H=((t-1)%10)+1.
# Each job sources fir_env.sh (StdEnv/2023+python/3.11+scipy-stack+cuda/12.6+cudnn
# + .venv_fir). Underpowered (H>H_max) tasks are auto-skipped by the runner.
#
# Run ON fir (login-safe; only sbatch):
#   ssh fir 'cd /scratch/sesma/amica-python/scripts/cc_benchmark && bash submit_multimodel_hsweep.sh'
set -euo pipefail

CCDIR=/scratch/sesma/amica-python/scripts/cc_benchmark
OUT=/scratch/sesma/multimodel_bench/ds004505
mkdir -p "$OUT" "$CCDIR/logs"
cd "$CCDIR"

COMMON="--n-components 16 --duration-sec 0 --resample 250 --max-iter 2000 --num-mix 3 --skip-underpowered --output-dir $OUT"

# ---- smoke gate: sub-01, H=2 ----
SMOKE=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=mm_smoke
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd $CCDIR
source fir_env.sh
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset ds004505 --subject 1 --num-models 2 $COMMON
EOF
)
echo "smoke job: $SMOKE"

# ---- ds004505 H-sweep array (25 subjects x H=1..10), gated on smoke ----
sbatch --dependency=afterok:$SMOKE <<EOF
#!/bin/bash
#SBATCH --job-name=mm_ds004505_hsweep
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-250%30
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd $CCDIR
source fir_env.sh
T=\$SLURM_ARRAY_TASK_ID
SID=\$(( (T-1)/10 + 1 ))
H=\$(( (T-1)%10 + 1 ))
echo "task \$T -> subject \$SID, H \$H"
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset ds004505 --subject \$SID --num-models \$H $COMMON
EOF

echo "===== submitted; queue ====="
squeue --me -o "%.14i %.20j %.16a %.9T %.6D %R"
