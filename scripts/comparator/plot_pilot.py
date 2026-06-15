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

    # Ensure `from amica_python.benchmark.viz import paper_figures` works even
    # when this script is run without a `pip install -e .`.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # amica-python repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from amica_python.benchmark.viz import paper_figures as pf  # type: ignore

    bench_df = pd.read_csv(bench_csv)
    print(f"loaded bench_df from {bench_csv} ({len(bench_df)} rows)")
    # Give the comparator methods distinct, stable colors so bars are
    # distinguishable (METHOD_COLORS only knows the AMICA-Python display names).
    pf.METHOD_COLORS.setdefault("amica_python_jax", "#1F4E79")
    pf.METHOD_COLORS.setdefault("amica_python_jax_chunked", "#5B9BD5")
    pf.METHOD_COLORS.setdefault("scott_huberty_torch", "#D77A00")
    pf.METHOD_COLORS.setdefault("pyamica_torch", "#2A9D8F")
    pf.METHOD_COLORS.setdefault("neuromechanist_numpy", "#7B3294")
    pf.METHOD_COLORS.setdefault("fortran_amica17", "#8C8C8C")

    # Device-honest runtime + memory figure (reads the actual device column).
    rtm = pf.plot_comparator_runtime_memory(bench_df, out_dir, captions_dir)
    print(f"runtime+memory figure: {rtm}")

    if parity_csv.exists():
        parity_df = pd.read_csv(parity_csv)
        print(f"loaded parity_df from {parity_csv} ({len(parity_df)} rows)")
        par_paths = pf.plot_comparator_W_parity(parity_df, out_dir, captions_dir)
        print(f"W-parity figure: {par_paths}")
    else:
        print(f"(skipped W-parity; no {parity_csv})")


if __name__ == "__main__":
    main()
