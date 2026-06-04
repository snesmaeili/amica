# Reproducing the paper figures

This document covers end-to-end reproduction of every figure in the
*amica-python* zenodo preprint, from a clean checkout through to the PDFs
that go into `overleaf-paper/figures/`.

**Pre-built figures** are committed to `overleaf-paper/figures/` — the paper
compiles today without running anything. The steps below describe how to
regenerate them from source data.

---

## Quick reference: figure → script map

| Fig | File | Generating script | Input data |
|-----|------|-------------------|-----------|
| 1 — AMICA workflow | `fig_amica_workflow.png` | Static PNG (manually created) | — |
| 2 — Software architecture | `fig_software_architecture.pdf` | `scripts/zenodo_figures/fig_software_architecture.dot` | — |
| 3 — Synthetic recovery | `fig_synthetic_recovery.pdf` | `scripts/zenodo_figures/render_fig_synthetic_recovery.py` | `synthetic_long_all_metrics.csv` |
| 4 — MNE sample topomaps | `fig_mne_sample_topomaps.pdf` | `scripts/mne_sample_demo/run_mne_sample_demo.py` | mne.datasets.sample (auto-download) |
| 5 — MIR combined | `fig_mir_combined.pdf` | `scripts/zenodo_figures/render_fig_mir_combined.py` | `benchmark_results.csv` |
| 6 — Cumulative dipolarity | `fig01_cumulative_dipolarity.pdf` | `scripts/zenodo_figures/render_cluster_figures.py` | `benchmark_results.csv` + `component_metrics.csv` |
| 7 — Quality-cost / runtime | `fig_quality_cost.pdf` | `scripts/zenodo_figures/render_fig_runtime_combined.py` | `benchmark_results.csv` |
| 8 — Convergence trajectory | `fig07_amica_iterations.pdf` | `scripts/zenodo_figures/render_cluster_figures.py` | `benchmark_results.csv` + `iteration_trace.csv` |
| S1 — Backend parity | `fig_backend_parity.pdf` | `scripts/zenodo_figures/render_cluster_figures.py` | `benchmark_results.csv` + `component_metrics.csv` |
| S2 — Kappa diagnostic | `fig08_kappa_sufficiency.pdf` | `scripts/zenodo_figures/render_cluster_figures.py` | `benchmark_results.csv` |

The three CSV files in the "Input data" column are produced by running the
benchmark pipeline (Steps 4–7 below). They are **not committed to the
repository** because of size; the committed figures were rendered from the
`v3_paper_stage1_cluster` cluster run.

---

## What you need

| Resource | Required for |
|----------|-------------|
| Python ≥ 3.10 with `amica-python[all]` | Everything |
| graphviz (`dot` command) | Fig 2 only |
| GPU with JAX/CUDA (paper used NVIDIA H100) | Paper-exact GPU results; CPU path is ~200× slower |
| ds004505 OpenNeuro dataset (~10 GB, 25 subjects) | Figs 5–8, S1–S2 |
| mne.datasets.sample (~1.5 GB, auto-downloads) | Fig 4 |
| SLURM cluster (optional) | Parallel 25-subject runs; submit scripts in `scripts/cc_benchmark/` |

---

## Step 0 — Install

```bash
git clone https://github.com/snesmaeili/amica-python.git
cd amica-python

# CPU-only (figures 2 and 4; rendering from existing CSVs):
pip install -e ".[all]"

# GPU (paper-exact AMICA runs; requires CUDA 12):
pip install -e ".[all]" "jax[cuda12]>=0.4"
```

Verify:

```bash
python -c "import amica_python; print(amica_python)"
python -c "import jax; print(jax.devices())"   # should show gpu device
```

---

## Step 1 — Fig 2: software architecture (seconds, no data needed)

Requires graphviz:

```bash
dot -Tpdf scripts/zenodo_figures/fig_software_architecture.dot \
    -o overleaf-paper/figures/fig_software_architecture.pdf
```

---

## Step 2 — Fig 4: MNE sample demo (~10 min, CPU, ~1.5 GB download)

Reproduces the MNE-sample topomap figure and the same-seed reproducibility
metrics reported in §results-mne-sample. MNE sample data downloads
automatically on first run.

```bash
python scripts/mne_sample_demo/run_mne_sample_demo.py \
    --out-dir scripts/mne_sample_demo/results \
    --n-components 20 \
    --max-iter 3000
```

Copy output to overleaf:

```bash
cp scripts/mne_sample_demo/results/fig_mne_sample_topomaps.pdf \
   overleaf-paper/figures/fig_mne_sample_topomaps.pdf
```

---

## Step 3 — Fig 3: render synthetic figure from existing CSV (seconds)

If you already have `synthetic_long_all_metrics.csv` from a prior run:

```bash
python scripts/zenodo_figures/render_fig_synthetic_recovery.py \
    --csv scripts/mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv \
    --out overleaf-paper/figures/fig_synthetic_recovery.pdf
```

If you need to produce the CSV from scratch, proceed to Step 5 below.

---

## Step 4 — Download ds004505 (~10 GB)

Required for Figs 5–8, S1–S2.

```bash
pip install openneuro-py
python -c "
import openneuro
openneuro.download('ds004505', target_dir='/your/path/ds004505')
"
export BIDS_ROOT_DS4505=/your/path/ds004505
```

> **Note — `--input-level`**: the value depends on what was downloaded.
>
> | Layout on disk | Flag to use |
> |---|---|
> | `sub-XX/eeg/*.set` (standard BIDS) | `--input-level bids` |
> | `sourcedata/Merged/sub-XX/*.set` (openneuro-py default) | `--input-level merged` |
> | try BIDS first, fall back to Merged | `--input-level auto` (default) |
>
> The paper cluster run used `--input-level bids` (pre-processed BIDS files on
> the Alliance Canada filesystem). A fresh `openneuro.download` produces the
> `sourcedata/Merged/` layout, so use `--input-level merged` in that case.

---

## Step 5 — Run the real-EEG benchmark (Figs 5–8, S1–S2)

### Option A — local, one subject at a time (GPU recommended)

Run all 25 subjects × all backends (paper used H100 for JAX-GPU; CPU
backends take ~7–8 hours per subject). Replace `--input-level merged` with
`bids` or `auto` to match your download layout (see note in Step 4).

**Single-subject test first (recommended before running all 25):**

```bash
# JAX-GPU, sub-01 only
BIDS_ROOT_DS4505=/your/path/ds004505 \
AMICA_COMPUTE_DIPOLES=0 \
  python -m amica_python.benchmark.runner \
    --dataset ds004505 \
    --subject 1 \
    --backend jax \
    --device gpu \
    --n-iter 3000 \
    --input-level merged \
    --schema-version v3 \
    --output-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
```

**All 25 subjects:**

```bash
# JAX-GPU backend (paper-exact; needs CUDA)
# Single subject: replace $(seq 1 25) with a single number, e.g. --subject 1
for i in $(seq 1 25); do
  AMICA_COMPUTE_DIPOLES=1 \
    python -m amica_python.benchmark.runner \
      --dataset ds004505 \
      --subject $i \
      --backend jax \
      --device gpu \
      --n-iter 3000 \
      --input-level merged \
      --schema-version v3 \
      --output-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
done

# JAX-CPU backend
# Single subject: --subject 1
for i in $(seq 1 25); do
  AMICA_COMPUTE_DIPOLES=1 \
    python -m amica_python.benchmark.runner \
      --dataset ds004505 \
      --subject $i \
      --backend jax \
      --device cpu \
      --n-iter 3000 \
      --input-level merged \
      --schema-version v3 \
      --output-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
done

# NumPy-CPU backend
# Single subject: --subject 1
for i in $(seq 1 25); do
  AMICA_COMPUTE_DIPOLES=0 \
    python -m amica_python.benchmark.runner \
      --dataset ds004505 \
      --subject $i \
      --backend numpy \
      --device cpu \
      --n-iter 3000 \
      --input-level merged \
      --schema-version v3 \
      --output-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
done

# Comparators: Picard, Infomax, FastICA
# Single subject example: --subject 1 --method picard
for method in picard infomax fastica; do
  for i in $(seq 1 25); do
    AMICA_COMPUTE_DIPOLES=1 \
      python -m amica_python.benchmark.runner \
        --dataset ds004505 \
        --subject $i \
        --method $method \
        --input-level merged \
        --schema-version v3 \
        --output-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
  done
done
```

Smoke test (MNE sample data, no ds004505 download needed, ~2 min):

```bash
AMICA_COMPUTE_DIPOLES=0 \
  python -m amica_python.benchmark.runner \
    --dataset mne \
    --backend jax \
    --device gpu \
    --n-iter 3000 \
    --schema-version v3 \
    --output-dir ./local_results
```

### Option B — SLURM cluster (paper method)

```bash
cd scripts/cc_benchmark

# Edit fir_env.sh to set your module loads, venv path, and BIDS_ROOT_DS4505.
# Then submit one array job per backend:
sbatch submit_jax_gpu_v3.sh        # 25-subject array, H100, JAX-GPU
sbatch submit_jax_cpu_v3.sh        # 25-subject array, JAX-CPU
sbatch submit_numpy_cpu_v3.sh      # 25-subject array, NumPy-CPU
sbatch submit_picard_cpu_v3.sh     # Picard comparator
sbatch submit_infomax_cpu_v3.sh    # Extended Infomax comparator
sbatch submit_fastica_cpu_v3.sh    # FastICA comparator
```

Results land in `$AMICA_RESULTS_DIR` (default: `$SCRATCH/amica_python_validation_v3`).
Copy the results directory to
`scripts/cc_benchmark/results/v3_paper_stage1_cluster/` before aggregating.

---

## Step 6 — Aggregate benchmark JSONs → CSVs

```bash
python scripts/cc_benchmark/aggregate_results.py \
    --results-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster
```

Produces three files in that directory:
- `benchmark_results.csv` — one row per (subject, method); MIR, PMI, runtime, κ, …
- `component_metrics.csv` — one row per (subject, method, component); dipolarity, ICLabel, …
- `iteration_trace.csv` — one row per (subject, method, iteration); log-likelihood trajectory

---

## Step 7 — Render Figs 5, 6, 7, 8, S1, S2 from CSVs

### Figs 5, 7 via make_all.sh (Fig 3 synthetic also included if CSV exists)

```bash
OUT_DIR=$(pwd)/overleaf-paper/figures \
  bash scripts/zenodo_figures/make_all.sh
```

This writes `fig_synthetic_recovery.pdf`, `fig_mir_combined.pdf`, and
`fig_quality_cost.pdf` directly to the overleaf figures directory.

### Figs 6, 8, S1, S2 via render_cluster_figures.py

```bash
AMICA_NO_RUN_MODE_BANNER=1 \
  python scripts/zenodo_figures/render_cluster_figures.py \
    --results-dir scripts/cc_benchmark/results/v3_paper_stage1_cluster \
    --out overleaf-paper/figures
```

Writes:
- `fig01_cumulative_dipolarity.pdf` (Fig 6)
- `fig07_amica_iterations.pdf` (Fig 8)
- `fig08_kappa_sufficiency.pdf` (S2)
- `fig_backend_parity.pdf` (S1)

---

## Step 8 — Run the synthetic benchmark (Fig 3)

Required only if `synthetic_long_all_metrics.csv` does not yet exist.

### 8a — Generate one (condition, seed, method) fit

```bash
python scripts/mne_synthetic/run_one_synthetic.py \
    --config scripts/mne_synthetic/configs/benchmark_v1.json \
    --condition clean \
    --seed 101 \
    --method numpy_cpu \
    --results-dir scripts/mne_synthetic/results/v1_full_analysis
```

### 8b — Run all 300 fits (6 methods × 5 conditions × 10 seeds)

Local (sequential, slow — GPU methods take minutes each, CPU methods longer):

```bash
for method in jax_gpu jax_cpu numpy_cpu picard infomax fastica; do
  for condition in clean noise noise_eog noise_ecg full; do
    for seed in 101 202 303 404 505 606 707 808 909 1010; do
      python scripts/mne_synthetic/run_one_synthetic.py \
        --config scripts/mne_synthetic/configs/benchmark_v1.json \
        --condition $condition \
        --seed $seed \
        --method $method \
        --results-dir scripts/mne_synthetic/results/v1_full_analysis
    done
  done
done
```

SLURM (50-task array per method):

```bash
cd scripts/mne_synthetic
# Edit fir_env_synthetic.sh for your cluster environment, then:
sbatch submit_jax_gpu_synthetic.sh
sbatch submit_jax_cpu_synthetic.sh
sbatch submit_numpy_cpu_synthetic.sh
sbatch submit_picard_synthetic.sh
sbatch submit_infomax_synthetic.sh
sbatch submit_fastica_synthetic.sh
```

### 8c — Aggregate synthetic JSONs → CSV

```bash
python scripts/mne_synthetic/aggregate_synthetic.py \
    --results-dir scripts/mne_synthetic/results/v1_full_analysis \
    --output-dir  scripts/mne_synthetic/results/v1_full_analysis
```

Produces `synthetic_long_all_metrics.csv` in that directory.

### 8d — Render Fig 3

```bash
python scripts/zenodo_figures/render_fig_synthetic_recovery.py \
    --csv scripts/mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv \
    --out overleaf-paper/figures/fig_synthetic_recovery.pdf
```

---

## Running on a local consumer GPU (not just H100)

AMICA-Python runs on a normal NVIDIA laptop/desktop GPU. Two knobs make the
difference between an OOM and a clean run:

### 1. `--dtype float32` (roughly halves memory, often much faster on consumer GPUs)

```bash
python -m amica_python.benchmark.runner --dataset ds004505 --subject 1 \
    --backend jax --device gpu --dtype float32 --chunk-size auto \
    --n-iter 2000 --schema-version v3
```

`float64` is the **reference / parity** mode and stays the default. `float32` is a
*fast* mode: it halves buffer memory and is markedly quicker on consumer GPUs that
have weak FP64 throughput. Validate float32 results against a float64 run before
trusting them for science (the package's tests assert float32 W matches float64 W
to matched mean |r| > 0.999 on a fixed seed).

### 2. `--chunk-size auto` is now VRAM-aware

`chunk_size="auto"` sizes the E-step chunk against the **active device's free
VRAM** when running on GPU (via `jax.devices()[0].memory_stats()`), and against
system RAM via psutil on CPU. It no longer OOMs on a small GPU. You can still pass
an explicit int to cap memory yourself:

| GPU VRAM | auto budget (¼ of free) | resulting chunk (64 comp, 3 mix, f64) |
|----------|-------------------------|----------------------------------------|
| 8 GB | ~2 GB | ~200,000 |
| 24 GB | ~4 GB (capped) | ~400,000 |
| 80 GB (H100) | ~4 GB (capped) | ~400,000, or full-batch if it fits |

`float32` doubles each chunk for the same budget.

### 3. XLA memory env vars (for tight-VRAM machines / OOM debugging)

JAX preallocates ~75% of VRAM by default. On a shared or small GPU:

```bash
# Cap JAX's preallocation to leave headroom for the OS / display:
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Or disable preallocation entirely while debugging an OOM (slower, less
# fragmentation-prone):
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Reading the timing fields

v3 JSON now splits one-time compile from steady-state cost:

- `jit_compile_s` — first-iteration XLA trace+compile. With the persistent
  compilation cache warm (same shape/dtype re-run), this drops to ≈ 0.
- `steady_iter_s` — median wall-time of a steady-state iteration (iters 1..N).

So a run that looks "slow" end-to-end is often compile-dominated on the first
shape; `steady_iter_s × n_iter` is the number that scales.

### Reproducibility note (float64 trajectory)

The solver skips a redundant per-iteration matrix inverse in the column-scaling
step by using the exact identity `inv(A·diag(1/c)) = diag(c)·inv(A)` instead of
recomputing `pinv`. This is exact in real arithmetic and passes the full parity
suite (reconstruction `frob_rel < 1e-12`, chunked vs full-batch `rel_err < 1e-4`),
but it is **not bit-reproducible** against older builds: the per-step difference
sits at the float64 SVD floor (~1e-10) and the EM loop amplifies it to ~1e-4 in
the final unmixing matrix over a few hundred iterations. This is the same class
of non-determinism as a BLAS/LAPACK version change — both runs are valid AMICA
fixed points at the same log-likelihood. Use a fixed seed and a pinned jaxlib if
you need run-to-run bitwise identity.

---

## Notes on `AMICA_COMPUTE_DIPOLES`

Setting `AMICA_COMPUTE_DIPOLES=0` skips the BEM dipole fitting step. This
is much faster but omits dipolarity metrics (Fig 6, S1). Set to `1` (default)
for paper-exact runs — requires `mne-icalabel` and takes extra time per subject.

```bash
# Fast smoke test (no dipolarity):
AMICA_COMPUTE_DIPOLES=0 python -m amica_python.benchmark.runner ...

# Paper-exact (with dipolarity):
AMICA_COMPUTE_DIPOLES=1 python -m amica_python.benchmark.runner ...
```

---

## Runtime expectations

| Backend | Dataset | Time per subject | Notes |
|---------|---------|-----------------|-------|
| JAX-GPU (H100) | ds004505, 64 comp, 3000 iter | ~134 s | Paper hardware |
| JAX-GPU (RTX 4070) | MNE sample, 20 comp | ~25 s | Local smoke test |
| JAX-CPU | ds004505, 64 comp, 3000 iter | ~7 h | Paper result |
| NumPy-CPU | ds004505, 64 comp, 3000 iter | ~8 h | Paper result |
| Picard (CPU) | ds004505, 64 comp | ~150 s | Converges ~69 iter |
| Infomax (CPU) | ds004505, 64 comp | ~84 s median | Heavy right tail |
| FastICA (CPU) | ds004505, 64 comp | ~99 s | Converges <5000 iter |

Reduce components with a code edit in `amica_python/benchmark/runner.py` if
you run out of GPU memory (`n_components` line in `run_benchmark`).

---

## Verifying a completed run

After aggregation, spot-check the CSVs:

```bash
python -c "
import pandas as pd
df = pd.read_csv('scripts/cc_benchmark/results/v3_paper_stage1_cluster/benchmark_results.csv')
print(df.groupby('method')[['mir_kbits_s','runtime_s']].median().round(3))
"
```

Expected output (matches Table 2 in the paper):

```
                             mir_kbits_s  runtime_s
method
amica_jax_gpu                      ~4.7      ~134
amica_jax_cpu                      ~4.7    ~25000
amica_numpy_cpu                    ~4.7    ~28000
fastica                            ~4.2       ~99
infomax                            ~4.2       ~84
picard                             ~4.3      ~150
```
