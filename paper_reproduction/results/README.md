# Results directory

This directory holds benchmark outputs. Large files (.pkl, .fdt) are .gitignored.

## Expected structure after a full run

```
results/
├── benchmark_report/       # CPU/GPU per-iteration timings (JSON per seed+device)
├── figures/                # Exploratory diagnostic figures
├── figures_paper/          # Publication-ready figures (PDF + PNG)
├── clean_eeg_test/         # MNE sample validation outputs
├── mne_chunking/           # Chunking parity validation
├── mne_endtoend/           # MNE pipeline integration test
├── eeglab_amica/           # EEGLAB cross-comparison outputs
├── post_f1_audit/          # Fortran audit verification
├── iclean_dual/            # Dual-layer preprocessing results
├── benchmark_sub-*.json    # Per-subject algorithm comparison metrics
├── quick_check_results.json
├── validation_results.json
└── PAPER_NOTES.md          # See docs/PAPER_NOTES.md for latest
```

## Regenerating results

```bash
# Quick single-subject check
make real-eeg

# Full 25-subject GPU comparison
make slurm-comparison

# CPU/GPU benchmarks
make slurm-benchmark

# Paper figures (after results are ready)
make paper-all
```
