# `scripts/zenodo_figures/`

Figure-generation scripts for the zenodo preprint
*`amica-python`: Native Python/JAX AMICA for MNE-compatible EEG Source Separation
— Validation and Benchmark Companion*.

These scripts produce three of the six main-text figures from already-aggregated
benchmark CSVs:

| Main figure | Script | Source CSV |
|---|---|---|
| Fig 1 — Synthetic ground-truth recovery | [`render_fig_synthetic_recovery.py`](render_fig_synthetic_recovery.py) | `scripts/mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv` |
| Fig 4 — MIR on ds004505 (combined) | [`render_fig_mir_combined.py`](render_fig_mir_combined.py) | `scripts/cc_benchmark/results/v3_paper_stage1_cluster/benchmark_results.csv` |
| Fig 5 — Quality–cost + runtime distribution | [`render_fig_runtime_combined.py`](render_fig_runtime_combined.py) | same as Fig 4 |

The remaining main-text figures are produced by other scripts in this repo
(see the table at the bottom of this README).

## One-shot regeneration

```bash
# Linux / macOS / cluster login (no compute on login — these scripts run
# locally from already-aggregated CSVs in seconds, no GPU/CUDA needed):
./make_all.sh

# Windows PowerShell:
.\make_all.ps1
```

To write into an Overleaf clone instead of `./figures/`:

```bash
OUT_DIR=/path/to/overleaf_repo/figures ./make_all.sh
```

```powershell
$env:OUT_DIR = "D:\overleaf_repos\<project-id>\figures"
.\make_all.ps1
```

Each renderer also writes a companion `<figure>_stats.csv` so the visual can be
re-derived from the source data.

## Style

All figures share the palette + rcParams in [`style.py`](style.py). The
method colour mapping is:

| Method | Hex |
|---|---|
| `amica-python` JAX-GPU | `#08306b` |
| `amica-python` JAX-CPU | `#2171b5` |
| `amica-python` NumPy-CPU | `#6baed6` |
| Picard | `#2ca02c` |
| Infomax | `#d62728` |
| FastICA | `#9467bd` |

For the synthetic benchmark, the AMICA iteration-cap variants are coloured as:

| Variant | Hex |
|---|---|
| `AMICA@3k` | `#6baed6` |
| `AMICA@10k` | `#08306b` |

## Dependencies

```
numpy
pandas
scipy
matplotlib
```

No JAX, no MNE, no GPU. These scripts consume already-fitted benchmark CSVs.
Re-generating the underlying CSVs requires the full benchmark pipeline
(`scripts/cc_benchmark/` and `scripts/mne_synthetic/`); see those subdirs.

## Where each main-text figure comes from

| Fig | Script / repo location |
|---|---|
| 1 — Synthetic recovery | `scripts/zenodo_figures/render_fig_synthetic_recovery.py` |
| 2 — MNE sample topomaps | `scripts/mne_sample_demo/run_mne_sample_demo.py` |
| 3 — Cumulative dipolarity | `scripts/cc_benchmark/paper_figures.py::plot_cumulative_dipolarity` |
| 4 — MIR combined | `scripts/zenodo_figures/render_fig_mir_combined.py` |
| 5 — Quality–cost + runtime | `scripts/zenodo_figures/render_fig_runtime_combined.py` |
| 6 — AMICA convergence | `scripts/cc_benchmark/paper_figures.py::plot_amica_iterations` |
| S1 — Backend parity | `scripts/cc_benchmark/paper_figures.py::plot_backend_parity` |
| S2 — Cohort κ diagnostic | `scripts/cc_benchmark/paper_figures.py::plot_kappa_sufficiency` |
