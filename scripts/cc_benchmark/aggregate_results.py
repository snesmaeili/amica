#!/usr/bin/env python
"""
Aggregate AMICA benchmark results from JSON files into summary tables.

Usage:
  python aggregate_results.py
  python aggregate_results.py --results-dir /path/to/results
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate AMICA benchmark results")
    parser.add_argument(
        "--results-dir", type=str,
        default=os.environ.get("AMICA_RESULTS_DIR", "results"),
        help="Directory containing JSON result files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_data = []

    for f in sorted(results_dir.glob("*.json")):
        with open(f) as f_in:
            data = json.load(f_in)
            if isinstance(data, dict):
                data = [data]
            all_data.extend(data)

    if not all_data:
        print(f"No results found in {results_dir}")
        return

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} benchmark results from {results_dir}\n")

    # ── Performance summary ──
    perf_cols = ["runtime_s", "n_iter", "n_components", "n_samples"]
    existing_perf = [c for c in perf_cols if c in df.columns]

    summary = df.groupby(["dataset", "backend", "device"])[existing_perf].agg(["mean", "std", "count"])
    print("## Performance Summary (mean ± std)\n")
    print(summary.to_string())
    print()

    # ── Speedup table ──
    if "runtime_s" in df.columns:
        pivot = df.groupby(["dataset", "backend", "device"])["runtime_s"].mean().reset_index()
        print("## Mean Runtime per Configuration\n")
        print(pivot.to_string(index=False))
        print()

    # ── ICLabel summary (if available) ──
    iclabel_cols = [c for c in df.columns if c.startswith("iclabel_")]
    if iclabel_cols:
        print("## ICLabel Component Classification\n")
        icl_summary = df.groupby(["backend", "device"])[iclabel_cols].mean()
        print(icl_summary.to_string())
        print()

    # ── Save to files ──
    csv_path = results_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to {csv_path}")

    md_path = results_dir / "benchmark_summary.md"
    with open(md_path, "w") as f_out:
        f_out.write("# AMICA Benchmark Results\n\n")
        f_out.write("## Performance Summary\n\n")
        f_out.write(summary.to_markdown() + "\n\n")
        if iclabel_cols:
            f_out.write("## ICLabel Summary\n\n")
            icl_summary = df.groupby(["backend", "device"])[iclabel_cols].mean()
            f_out.write(icl_summary.to_markdown() + "\n")
    print(f"Markdown report saved to {md_path}")


if __name__ == "__main__":
    main()
