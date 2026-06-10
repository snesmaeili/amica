"""Render comparator-pilot paper figures from the aggregator CSVs.

Consumes:
  results/comparison/comparator_pilot_bench.csv    (bench_df-shape; produced by aggregate_comparator_pilot.py)
  results/comparison/comparator_pilot_parity.csv   (subject x reference x compared x matched_mean_abs_corr)

Calls the in-tree figure functions:
  - paper_figures.plot_runtime_summary(bench_df, ...) - reused, no change
  - paper_figures.plot_comparator_W_parity(parity_df, ...) - new, added in this branch

Usage:
  python scripts/comparison/plot_comparator_pilot.py \
      --root results/comparison [--out-dir results/comparison/paper_figures_pilot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/comparison",
        help="Directory containing comparator_pilot_bench.csv + comparator_pilot_parity.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write figures (default: <root>/paper_figures_pilot)",
    )
    parser.add_argument(
        "--captions-dir",
        default=None,
        help="Where to write caption .txt files (default: <out_dir>/captions)",
    )
    parser.add_argument(
        "--bench-prefix",
        default="comparator_pilot",
        help="Filename prefix used by the aggregator",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "paper_figures_pilot")
    captions_dir = Path(args.captions_dir) if args.captions_dir else (out_dir / "captions")
    out_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    bench_csv = root / f"{args.bench_prefix}_bench.csv"
    parity_csv = root / f"{args.bench_prefix}_parity.csv"
    if not bench_csv.exists():
        raise SystemExit(
            f"missing {bench_csv}. Run aggregate_comparator_pilot.py first."
        )

    # Make the in-tree paper_figures importable when this script runs from the
    # benchmark repo without amica-python being on the orchestrator's path.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    amica_python_dir = repo_root.parent / "amica-python"
    if amica_python_dir.exists():
        sys.path.insert(0, str(amica_python_dir))

    from amica_python.benchmark.viz import paper_figures as pf  # type: ignore

    bench_df = pd.read_csv(bench_csv)
    print(f"loaded bench_df from {bench_csv} ({len(bench_df)} rows)")
    rt_paths = pf.plot_runtime_summary(bench_df, out_dir, captions_dir)
    print(f"runtime summary: {rt_paths}")

    if parity_csv.exists():
        parity_df = pd.read_csv(parity_csv)
        print(f"loaded parity_df from {parity_csv} ({len(parity_df)} rows)")
        par_paths = pf.plot_comparator_W_parity(parity_df, out_dir, captions_dir)
        print(f"W-parity figure: {par_paths}")
    else:
        print(f"(skipped W-parity; no {parity_csv})")


if __name__ == "__main__":
    main()
