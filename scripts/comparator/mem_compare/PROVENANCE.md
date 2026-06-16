# Cross-implementation memory comparison — provenance

Source of the memory figure + table in the preprint (`zenodo.tex`, Fig. `fig:mem_comparison`,
Table `tab:mem_comparison`).

## What
Peak memory of the publicly available AMICA implementations on one recording:
**ds004505 sub-01**, `C=64` PCA components, `T=785,328` samples (full 52-min recording), 100 iterations.

## Cluster runs (fir / Compute Canada)
| Job | What | Output |
|---|---|---|
| `44420385` | CPU comparator, 6 impls (amica full-batch + chunked, pyamica, scott, neuromechanist, Fortran) | `def-kjerbi`, 8 CPU, 40 G |
| `44423654` | GPU VRAM head-to-head (amica auto-chunk, pyamica, scott on one H100) | `def-kjerbi_gpu`, `gpu:h100:1` |
| `44424365` | Fortran recovery (after the `do_sphere=1/doPCA=1` fix) | `def-kjerbi_cpu` |

Code: `scripts/comparator/` at commit `94248d4682` (CPU+GPU jobs) and `b6d2251329` (Fortran fix,
`run_fortran.py do_sphere=1`). Orchestrator `implementation_perf.py`; runners `runners/run_*.py`.

## Measurement
- **Host RSS** = process high-water mark via `resource.getrusage(RUSAGE_SELF).ru_maxrss` (captures
  XLA/BLAS arenas, unlike `tracemalloc`). `baseline_rss_gb` is captured pre-fit; `delta_rss_gb` =
  peak − baseline (the fit's marginal footprint). The ~2.4 GB baseline is a shared data-load +
  framework-import floor common to the four Python implementations (Fortran, a separate binary
  measured via `/usr/bin/time -v`, has no such floor → baseline 0).
- **GPU VRAM** = each framework's allocator high-water mark with preallocation/caching disabled:
  XLA `peak_bytes_in_use` (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) for the JAX backend;
  `torch.cuda.max_memory_allocated` (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) for the PyTorch ones.

## Like-for-like check (Hungarian-matched unsigned |r| on the unmixing rows, this run)
amica_python_jax ↔ amica_python_jax_chunked = **1.000**; ↔ pyamica = **0.998**; ↔ scott = **0.998**.
(All implementations recover the same spatial filters on this run.)

## Numbers
See `mem_comparison_table.csv`. CPU peak RSS: amica full-batch **11.4**, chunked **6.6**, scott
**5.7**, pyamica **2.4** (= the load floor; its fit Δ ≈ 0), Fortran **1.5** GB. GPU peak VRAM:
amica (auto-chunk) **3.1**, scott **4.8**, pyamica **19.9** GB.

## Caveats
Single subject, single config (n=1, 100 iter, one H100). `neuromechanist/pyAMICA` is omitted: it
raised `LinAlgError: Singular matrix` mid-optimisation on this data (its own numerical fragility),
not a measurement failure.

## Regenerate the figure
```
cd scripts/comparator/mem_compare && python plot_mem_comparison.py
```
