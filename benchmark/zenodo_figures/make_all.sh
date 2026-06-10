#!/usr/bin/env bash
# Regenerate every zenodo-preprint figure that this directory owns.
#
# Reads:
#   ../mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv
#   ../cc_benchmark/results/v3_paper_stage1_cluster/benchmark_results.csv
# Writes:
#   ./figures/fig_synthetic_recovery.pdf       (+ _stats.csv)
#   ./figures/fig_mir_combined.pdf             (+ _stats.csv)
#   ./figures/fig_quality_cost.pdf             (+ _stats.csv)
#
# To target a custom output dir (e.g. an Overleaf clone):
#   OUT_DIR=/path/to/overleaf/figures ./make_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${HERE}/figures}"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"
echo "[zenodo_figures] writing to ${OUT_DIR}"

"${PYTHON}" "${HERE}/render_fig_synthetic_recovery.py" \
    --out "${OUT_DIR}/fig_synthetic_recovery.pdf"

"${PYTHON}" "${HERE}/render_fig_mir_combined.py" \
    --out "${OUT_DIR}/fig_mir_combined.pdf"

"${PYTHON}" "${HERE}/render_fig_runtime_combined.py" \
    --out "${OUT_DIR}/fig_quality_cost.pdf"

echo "[zenodo_figures] DONE"
ls -la "${OUT_DIR}"/fig_synthetic_recovery.pdf \
       "${OUT_DIR}"/fig_mir_combined.pdf \
       "${OUT_DIR}"/fig_quality_cost.pdf
