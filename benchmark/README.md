# Benchmark & validation scripts (archive)

This directory holds the **research/validation tooling** used to produce the
benchmark results and paper figures for `amica-python`. It is **not needed to use
the package** — for that, see [`../examples/`](../examples/) and the top-level
[`README`](../README.md). These scripts are cluster- and dataset-specific
(Compute Canada / Alliance paths, OpenNeuro datasets) and are kept here for
reproducibility and reference.

| Subdir | Purpose |
|--------|---------|
| `cc_benchmark/` | Compute Canada Slurm jobs + runners for the real-EEG benchmarks (single-model parity, kappa/runtime sweeps, and the multi-model stationarity H-sweep on ds004505/ds004504), plus the metric drivers and plotting. |
| `mne_synthetic/` | Synthetic ground-truth benchmark: generate known sources, fit, and score recovery (model error, SIR, source-PDF KL). |
| `comparator/` | Cross-implementation comparison runners (amica-python vs other AMICA/ICA implementations). |
| `zenodo_figures/` | Rendering scripts for the archived paper/preprint figures. |

## Notes

- Paths and accounts inside these scripts are specific to the authors' cluster and
  must be adapted before reuse.
- Large result artifacts are **not** committed here (they live in the separate
  `amica-python-benchmark` repository / release artifacts).
- The reusable metrics live in the package itself under
  [`amica_python/benchmark/`](../amica_python/benchmark/) (e.g. `metrics.py`,
  `stationarity.py`, `dipolarity.py`); the scripts here orchestrate runs and
  figures around that API.
