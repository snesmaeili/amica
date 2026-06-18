#!/bin/bash
# Parameterized multi-model H-sweep for the Part C rework: extra cohorts + controls.
# Mirrors submit_multimodel_hsweep.sh (isolated clone /scratch/sesma/amica-mm + .venv_fir,
# smoke gate -> array). Drives run_multimodel_benchmark.py with per-site notch + the new
# --channel-subset / --surrogate hooks.
#
# Env vars:
#   DS          dataset (ds004504 | ds004621 | ds004505)
#   SUB_BASE    first subject id (e.g. 1 for ds004505/621, 37 for ds004504)
#   NSUB        number of subjects
#   OUTSUB      output subdir under /scratch/sesma/multimodel_bench/
#   JOBNAME     slurm job name
#   HMAX        max H (default 10); EXTRA   extra runner args (e.g. "--channel-subset tentwenty"
#               or "--surrogate phase"); INPUT_LEVEL (default bids); GATE (1=smoke-gate, default 1)
# Array = NSUB*HMAX; task t -> subject = SUB_BASE + (t-1)/HMAX, H = (t-1)%HMAX + 1.
#
# Run ON fir (login-safe; only sbatch), after the clone is on the multimodel-amica tip:
#   DS=ds004621 SUB_BASE=1 NSUB=25 OUTSUB=ds004621 JOBNAME=mm_ds004621 \
#     bash submit_multimodel_extra.sh
set -euo pipefail
: "${DS:?set DS}"; : "${SUB_BASE:?set SUB_BASE}"; : "${NSUB:?set NSUB}"
: "${OUTSUB:?set OUTSUB}"; : "${JOBNAME:?set JOBNAME}"
HMAX=${HMAX:-10}; EXTRA=${EXTRA:-}; INPUT_LEVEL=${INPUT_LEVEL:-bids}; GATE=${GATE:-1}

CCDIR=/scratch/sesma/amica-mm/scripts/cc_benchmark
OUT=/scratch/sesma/multimodel_bench/$OUTSUB
mkdir -p "$OUT" "$CCDIR/logs"
cd "$CCDIR"
COMMON="--n-components 16 --duration-sec 0 --resample 250 --max-iter 2000 --num-mix 3 --skip-underpowered --input-level $INPUT_LEVEL --output-dir $OUT $EXTRA"
NTASK=$((NSUB * HMAX))

DEP=""
if [ "$GATE" = "1" ]; then
  SMOKE=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}_smoke
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd $CCDIR
source fir_env.sh
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset $DS --subject $SUB_BASE --num-models 2 $COMMON
EOF
)
  echo "smoke job: $SMOKE"
  DEP="--dependency=afterok:$SMOKE"
fi

sbatch $DEP <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-${NTASK}%30
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd $CCDIR
source fir_env.sh
T=\$SLURM_ARRAY_TASK_ID
SID=\$(( $SUB_BASE + (T-1)/$HMAX ))
H=\$(( (T-1)%$HMAX + 1 ))
echo "task \$T -> subject \$SID, H \$H"
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset $DS --subject \$SID --num-models \$H $COMMON
EOF
echo "submitted $JOBNAME : $NTASK tasks (-> $OUT)"
