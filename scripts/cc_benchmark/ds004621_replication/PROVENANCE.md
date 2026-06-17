# ds004621 third-dataset MIR replication (high density) — provenance

High-density replication of the ds004505 MIR ranking (AMICA-Python vs Picard / extended Infomax /
FastICA) on the **128-channel** Nencki-Symfonia cohort, completing a three-dataset result spanning
19 / 64 / 128 channels (Overleaf commit `2a4da7b`, preprint §Results "Replication across datasets" +
Table `tab:replication` + `fig_ds004621_mir`).

## Dataset
- **ds004621** (Dzianok et al. 2022, *GigaScience*; Nencki-Symfonia EEG/ERP), eyes-closed
  resting-state (`task-rest`).
- **42 healthy young adults** (`sub-01 … sub-42`).
- **128 EEG channels** (10-5 layout; the FCz online reference is excluded, leaving 127 data channels),
  1000 Hz, ~12.2 min (≈732,700 samples) per subject. BrainVision (`.vhdr/.eeg/.vmrk`).

## Configuration (identical pipeline to the ds004505 v3 headline)
- `N = 64` PCA components (128-ch acquisition reduced to 64, matching the ds004505 scale).
- Preprocess: 1–100 Hz FIR band-pass + 60 Hz notch (`runner.preprocess`, dataset-agnostic).
- AMICA-JAX: 3000 iterations, H100 GPU. Comparators: 5000-iteration ceiling, early-stop at tolerance;
  `--random-state 42`.

## Cluster jobs (fir)
- AMICA GPU array: job `44680886`, `submit_jax_gpu_ds004621.sh`, array 1–42.
- Comparators CPU array: job `44680887`, `submit_comparators_ds004621.sh` (`--method all`), array 1–42.
- All 168 result JSONs in `/scratch/sesma/amica_ds004621_v3/`. Dataset staged at
  `/scratch/sesma/datasets/ds004621/` (task-rest only, via OpenNeuro S3).

## Reproduce
```bash
# fir (after git pull of jax-performance-pass; dataset staged):
sbatch scripts/cc_benchmark/submit_jax_gpu_ds004621.sh
sbatch scripts/cc_benchmark/submit_comparators_ds004621.sh
# locally, after rsync of the JSONs:
python -m amica_python.benchmark.aggregate          --results-dir <dir> --output-dir <dir>
python -m amica_python.benchmark.viz.paper_figures  --results-dir <dir> --out-dir <dir>/figures --headless
```
The ds004621 loader (`runner.load_data`, BrainVision branch) + both submit scripts are committed on
`jax-performance-pass` (`9c22c2e3cc`).

## Result (n = 42; full numbers in `fig09_paired_mir_stats.csv`)
Across-subject mean MIR (kbits/s): **AMICA 14.68**, Picard 12.45, Infomax 12.43, FastICA 12.34.

Paired ΔMIR (AMICA − competitor), positive for **42/42** subjects in every contrast:

| Contrast | mean ΔMIR (kbits/s) | d_z | Holm-adj t-test p | Wilcoxon p |
|---|---|---|---|---|
| vs Picard  | +2.229 | 1.48 | 1.47e-11 | 4.55e-13 |
| vs Infomax | +2.248 | 1.45 | 1.79e-11 | 4.55e-13 |
| vs FastICA | +2.337 | 1.44 | 1.79e-11 | 4.55e-13 |

The effect is large (d_z ≈ 1.45), between ds004504 (0.78, low-density clinical resting) and ds004505
(≈2.5, high-density task). Dipolarity (nd@10%) also favours AMICA (10.2% vs 8.2–8.8%). ICLabel is
unavailable for this cohort: 4 channels (the TP9/TP10 ear references and O9/O10) lack `standard_1005`
positions, so the ICLabel columns in `benchmark_results.csv` are NaN — MIR and dipolarity (the
reported metrics) are unaffected.

## Files
- `benchmark_results.csv` — one row per (subject, method); MIR, dipolarity, runtime (ICLabel NaN).
- `fig09_paired_mir_stats.csv` — per-contrast paired stats (mean, d_z, t/Wilcoxon/perm p, Holm).
- `fig_ds004621_mir.pdf` — the figure used in the preprint.
