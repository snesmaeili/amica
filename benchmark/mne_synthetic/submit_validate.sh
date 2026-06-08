#!/bin/bash
#SBATCH --job-name=synth_validate
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/synth_validate_%j.out
#SBATCH --error=/home/sesma/scratch/synth_validate_%j.err

# End-to-end validation: ONE jax_gpu task on (clean, 101) with --max-iter 200.
# Tests every layer (env script, all imports, JAX GPU detect, generate Raw,
# fit AMICA, score GT, write JSON) BEFORE we commit to the full 50-task array.
# Walltime 30 min is plenty; expected runtime is ~5 min total.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source fir_env_synthetic.sh

echo ""
echo "=== Step 1: import sanity check ==="
python -c "import nibabel; print('  nibabel:', nibabel.__version__)"
python -c "import picard; print('  python-picard imported OK')"
python -c "import mne; print('  mne:', mne.__version__)"
python -c "import amica_python; print('  amica_python imported OK')"
python -c "import jax; print('  jax:', jax.__version__); print('  devices:', jax.devices())"

echo ""
echo "=== Step 2: end-to-end fit (clean, seed=101, jax_gpu, max_iter=200) ==="
VALIDATE_DIR="/scratch/$USER/synth_validate_run"
rm -rf "$VALIDATE_DIR"
python run_one_synthetic.py \
    --config configs/benchmark_v1.json \
    --condition clean --seed 101 --method jax_gpu \
    --max-iter 200 \
    --results-dir "$VALIDATE_DIR"

echo ""
echo "=== Step 3: verify JSON output ==="
ls -la "$VALIDATE_DIR/"
python - << 'PYEOF'
import json, os, sys
path = f"/scratch/{os.environ['USER']}/synth_validate_run/synth_clean_seed-0101_jax_gpu.json"
doc = json.load(open(path))
fit = doc["jax_gpu"]
gt = fit["ground_truth"]
print(f"  runtime={fit['runtime_s']:.1f}s, n_iter={fit['n_iter']}/{fit['max_iter']}, converged={fit['converged_before_cap']}")
print(f"  GT: r_topo_median={gt['r_topo_median']:.4f}, r_topo_min={gt['r_topo_min']:.4f}, amari={gt['amari_index']:.4f}")
# Sanity checks
ok = True
if gt['r_topo_median'] < 0.5:
    print(f"  FAIL: r_topo_median {gt['r_topo_median']:.3f} < 0.5 (much worse than expected for clean condition)")
    ok = False
if gt['amari_index'] > 0.5:
    print(f"  FAIL: amari {gt['amari_index']:.3f} > 0.5 (much worse than expected for clean condition)")
    ok = False
if not ok:
    sys.exit(1)
print("  PASSED basic sanity (clean-condition GT is reasonable)")
PYEOF

echo ""
echo "=== VALIDATION COMPLETE ==="
