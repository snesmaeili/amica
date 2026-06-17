# ds004504 second-dataset MIR replication — provenance

Replication of the ds004505 MIR ranking (AMICA-Python vs Picard / extended Infomax / FastICA) on an
independent **eyes-closed resting** cohort, to test whether the ranking holds beyond the high-density
table-tennis task recording. Reported in the preprint §Results (MIR), "Replication on a second
dataset" (Overleaf commit `c5579c2`).

## Dataset
- **ds004504** (Miltiadous et al. 2023), eyes-closed resting EEG, `task-eyesclosed`.
- **29 healthy controls** — subjects `sub-037 … sub-065` (group `C`). These are the git-annex-
  materialized subset present on fir; no download was required.
- 19 channels (10-20 montage: Fp1…Pz), 500 Hz, ~13 min (≈388,550 samples) per subject.

## Configuration (identical pipeline to the ds004505 v3 headline)
- `N = 15` PCA components (under the 19-ch data rank; matches the `N16` precedent for this dataset).
- Preprocess: 1–100 Hz FIR band-pass + 60 Hz notch (`runner.preprocess`, dataset-agnostic).
- AMICA-JAX: 3000 iterations, H100 GPU. Comparators: shared 5000-iteration ceiling, each early-stops
  at its own tolerance; `--random-state 42`.
- Methods: `AMICA-Python (JAX-GPU)`, `Picard` (extended, ortho=False, tol 1e-6), `Infomax`
  (extended), `FastICA`.

## Cluster jobs (fir, Compute Canada)
- AMICA GPU array: job `44576962`, `submit_jax_gpu_ds004504.sh`, array 37–65.
- Comparators CPU array: job `44576963`, `submit_comparators_ds004504.sh` (`--method all`), array 37–65.
- All 58 array tasks COMPLETED; 116 result JSONs in `/scratch/sesma/amica_ds004504_v3/`.

## Reproduce
```bash
# on fir (after `git pull` of jax-performance-pass):
sbatch scripts/cc_benchmark/submit_jax_gpu_ds004504.sh
sbatch scripts/cc_benchmark/submit_comparators_ds004504.sh
# locally, after rsync of the JSONs:
python -m amica_python.benchmark.aggregate     --results-dir <dir> --output-dir <dir>
python -m amica_python.benchmark.viz.paper_figures --results-dir <dir> --out-dir <dir>/figures --headless
```
The ds004504 loader (`runner.load_data`, `comparators`) and both submit scripts are committed on
`jax-performance-pass` (`b9d2285a3a`). `aggregate.py` and `viz/paper_figures.py` are dataset-agnostic.

## Result (n = 29; full numbers in `fig09_paired_mir_stats.csv`)
Across-subject mean MIR (kbits/s): **AMICA 1.45**, Infomax 1.19, Picard 1.19, FastICA 1.16.

Paired per-subject ΔMIR (AMICA − competitor), positive for **29/29** subjects in every contrast:

| Contrast | mean ΔMIR (kbits/s) | d_z | Holm-adj t-test p | Wilcoxon p |
|---|---|---|---|---|
| vs Picard  | +0.258 | 0.78 | 7.3e-4 | 3.7e-9 |
| vs Infomax | +0.256 | 0.78 | 2.6e-4 | 3.7e-9 |
| vs FastICA | +0.287 | 0.78 | 5.0e-4 | 3.7e-9 |

The effect is smaller than on ds004505 (d_z ≈ 2.5) and the absolute MIR is lower — both expected at
19 ch / N=15 and on a lower-arousal resting paradigm — but the AMICA advantage is directionally
identical and statistically unambiguous. Secondary metrics (dipolarity, ICLabel brain %) are in
`benchmark_results.csv`; dipolarity does not separate the methods (consistent with ds004505), so it
is not claimed as a win.

## Files
- `benchmark_results.csv` — one row per (subject, method); MIR, dipolarity, ICLabel, runtime.
- `fig09_paired_mir_stats.csv` — per-contrast paired stats (mean, d_z, t/Wilcoxon/perm p, Holm).
- `fig_ds004504_mir.pdf` — the figure used in the preprint (= `fig04_mir_difference`).
