# Stage 3E — Fused E-step cluster re-comparison (AMICA-Python vs pyamica)

Head-to-head of AMICA-Python (Stage 3D fused E-step, branch `jax-performance-pass`)
against `pyamica` (DerAndereJohannes, PyTorch) on real EEG, re-running the pilot
comparison after the full-batch E-step was fused into one JIT graph.

Scripts: [`submit_fused_compare.sh`](submit_fused_compare.sh) (fir submission),
[`plot_fused_compare.py`](plot_fused_compare.py) (local analysis + figure).

## Configuration

- Dataset: ds004505, subjects **1, 2, 4** (sub-09 excluded — broken symlinks on the fir copy).
- Preprocessing: 10-min crop, resample 250 Hz, 1–100 Hz bandpass + 60 Hz notch, PCA → 64 components.
- **500 iterations, 1 seed** (seed=0). Matches the pilot for direct comparability.
- Cluster: **fir**. Modules via `scripts/cc_benchmark/fir_env.sh`
  (StdEnv/2023, python/3.11, scipy-stack, **cuda/12.6, cudnn**).
- Accounts: `def-kjerbi_cpu` (CPU); `def-kjerbi_gpu` / `gpubase_bygpu_b1` / `gpu:h100:1` (GPU).
- venvs: `.venv_fir` (AMICA + JAX-CUDA, editable), `.venv_competitors` (pyamica).
- AMICA E-step: `estep="auto"` → fused (default since Stage 3D).

## Slurm jobs

| Job | ID | Account | Notes |
|---|---|---|---|
| smoke (GPU, 50 iter) | 43154877 | def-kjerbi_gpu | device=gpu confirmed |
| fused_amica_cpu | 43157083 | def-kjerbi_cpu | subjects 1,2,4 |
| fused_amica_gpu | 43157084 | def-kjerbi_gpu | sub-02 hit a transient `cuda`-init failure |
| fused_amica_gpu (sub-02 re-run) | 43158662 | def-kjerbi_gpu | recovered sub-02 |
| fused_pyamica_cpu | 43157085 | def-kjerbi_cpu | subjects 1,2,4 |

## Results (500 iter)

Median across subjects:

| Series | runtime (median) | peak memory | converged LL |
|---|---|---|---|
| AMICA-classic CPU *(old pilot)* | 714 s | 0.46 GB | — |
| **AMICA-fused CPU** | **210 s** | 0.60 GB | matches pyamica |
| pyamica CPU *(fresh)* | 655 s | 1.38 GB | matches AMICA |
| **AMICA-fused GPU** | **4.2 s** | 3.03 GB | matches |

Per-subject runtime (s):

| subject | AMICA-fused CPU | pyamica CPU | AMICA-fused GPU |
|---|---|---|---|
| sub-01 | 170.5 | 602.9 | 4.21 |
| sub-02 | 234.9 | 654.9 | 20.86\* |
| sub-04 | 210.4 | 668.0 | 3.81 |

\* sub-02 GPU ran on a fresh node with a **cold JIT-compile cache**; the 20.86 s is
compile overhead, not slower compute — its converged LL (−0.9235) is identical to the
warm runs and to the CPU result. `fit_time_s` includes the first-iteration compile.

## Findings (preliminary — this benchmark configuration only)

- The fused E-step makes AMICA-Python **~3.4× faster on CPU** than the prior recompute path (714 → 210 s).
- AMICA-fused-CPU is **~3.1× faster than pyamica-torch on CPU**, at **~2.3× less memory** (0.60 vs 1.38 GB).
- AMICA-fused-GPU completes 500 iter in **~4 s** (~156× faster than pyamica-CPU).
- Converged log-likelihoods **match pyamica per subject** — equivalent solution quality.
- **This reverses the earlier pilot finding** (pyamica slightly faster on CPU: 685 vs 714 s).
  Requires further validation (more subjects, multi-seed, W-parity) before manuscript claims.

## Reproduce

```bash
# 1) submit on fir (login-safe; only calls sbatch)
ssh fir 'cd /scratch/$USER/amica-python && bash scripts/comparator/submit_fused_compare.sh'

# 2) rsync the 9 result JSONs locally (cluster is compute-only), then:
python scripts/comparator/plot_fused_compare.py \
    --new-root results/comparator/cmp_fused \
    --old-root results/comparator_pilot_cluster
```
