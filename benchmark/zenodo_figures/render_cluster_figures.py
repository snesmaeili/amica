"""Re-render the three cluster figures used by the zenodo preprint with the
PILOT/paper-mode banner suppressed.

The banner is a cluster-side safety mechanism (it tags every benchmark figure
with run-mode metadata so PILOT cropped runs cannot be confused with full
paper-mode runs); it should not appear in the final publication PDFs.

Inputs (default paths under the cc_benchmark results dir):
  scripts/cc_benchmark/results/v3_paper_stage1_cluster/
      benchmark_results.csv
      component_metrics.csv
      iteration_trace.csv

Outputs (default: ./figures/):
  fig01_cumulative_dipolarity.pdf
  fig07_amica_iterations.pdf
  fig08_kappa_sufficiency.pdf

Pass ``--out`` to redirect into the Overleaf clone.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_results = (
        here.parent / "cc_benchmark" / "results" / "v3_paper_stage1_cluster"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=default_results,
        help=f"Directory with benchmark CSVs (default: {default_results})",
    )
    parser.add_argument(
        "--out", type=Path, default=here / "figures",
        help="Output directory for the rendered PDFs",
    )
    args = parser.parse_args()

    # Suppress the run-mode banner for paper-grade renders.
    os.environ["AMICA_NO_RUN_MODE_BANNER"] = "1"

    # Import after env var is set so the figure code picks it up.
    from amica_python.benchmark.viz.paper_figures import (
        plot_amica_convergence,
        plot_cumulative_dipolarity,
        plot_data_sufficiency,
    )

    bench_df = pd.read_csv(args.results_dir / "benchmark_results.csv")
    comp_df = pd.read_csv(args.results_dir / "component_metrics.csv")
    iter_df = pd.read_csv(args.results_dir / "iteration_trace.csv")

    args.out.mkdir(parents=True, exist_ok=True)
    captions_dir = args.out / "_captions"
    captions_dir.mkdir(parents=True, exist_ok=True)

    p1, _ = plot_cumulative_dipolarity(comp_df, bench_df, args.out, captions_dir)
    print(f"WROTE {p1}")

    p2, _ = plot_amica_convergence(iter_df, args.out, captions_dir, bench_df=bench_df)
    print(f"WROTE {p2}")

    p3, _ = plot_data_sufficiency(bench_df, args.out, captions_dir)
    print(f"WROTE {p3}")


if __name__ == "__main__":
    main()
