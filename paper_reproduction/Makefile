.PHONY: help check-env parity comparison benchmark-cpu benchmark-gpu \
       sensitivity real-eeg paper-figures paper-tables paper-all \
       summarize slurm-parity slurm-comparison slurm-benchmark \
       slurm-real-eeg slurm-paper slurm-all clean

PYTHON ?= python
SBATCH ?= sbatch

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-env:  ## Verify JAX, amica-python, GPU availability
	$(PYTHON) -c "import amica_python; print(f'amica-python {amica_python.__version__}')"
	$(PYTHON) -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"
	$(PYTHON) -c "import mne; print(f'MNE {mne.__version__}')"

# ── Local runs ──────────────────────────────────────────────

parity:  ## Fortran parity checks (CPU, ~10 min)
	$(PYTHON) scripts/parity/validate_parity.py

comparison:  ## Algorithm comparison on MNE sample data
	$(PYTHON) scripts/comparison/run_validation.py

benchmark-cpu:  ## CPU performance benchmark (3 seeds)
	JAX_PLATFORMS=cpu $(PYTHON) scripts/performance/benchmark_report.py --device cpu

benchmark-gpu:  ## GPU performance benchmark (3 seeds)
	JAX_PLATFORMS=cuda $(PYTHON) scripts/performance/benchmark_report.py --device gpu

sensitivity:  ## Parameter sensitivity sweep
	$(PYTHON) scripts/sensitivity/run_rejection_ablation.py

real-eeg:  ## Quick real EEG validation (requires ds004505)
	$(PYTHON) scripts/real_eeg/quick_full_check.py

# ── Paper generation ────────────────────────────────────────

paper-figures:  ## Generate all publication figures
	$(PYTHON) scripts/paper/run_paper_pipeline.py --figures-only

paper-tables:  ## Generate summary tables
	$(PYTHON) scripts/paper/generate_paper_tables.py

paper-all: paper-figures paper-tables  ## Full paper pipeline

# ── Summaries ───────────────────────────────────────────────

summarize:  ## Compile all benchmark JSONs into report
	$(PYTHON) scripts/summarize.py

# ── Slurm submission ────────────────────────────────────────

slurm-parity:  ## Submit parity jobs to Narval
	$(SBATCH) slurm/parity/submit_parity_cpu.sh

slurm-comparison:  ## Submit 25-subject comparison job array
	$(SBATCH) slurm/comparison/submit_full_validation_gpu.sh

slurm-benchmark:  ## Submit CPU + GPU benchmark jobs
	$(SBATCH) slurm/performance/submit_benchmark_report_cpu.sh
	$(SBATCH) slurm/performance/submit_benchmark_report_gpu.sh

slurm-real-eeg:  ## Submit real EEG validation jobs
	$(SBATCH) slurm/real_eeg/submit_mne_endtoend.sh

slurm-paper:  ## Submit paper pipeline job
	$(SBATCH) slurm/paper/submit_topos_v2.sh

slurm-all: slurm-parity slurm-comparison slurm-benchmark slurm-real-eeg slurm-paper  ## Submit everything

# ── Utilities ───────────────────────────────────────────────

clean:  ## Remove results, logs, __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/
	@echo "To also clear results: rm -rf results/*"
