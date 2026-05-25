"""
Render the zenodo preprint quality-cost + runtime figure (main Fig 5).

Two-panel composite:
  Panel A (left, wider): quality-cost scatter: across-subject mean MIR
                         vs median fit runtime per method, log x. The
                         three AMICA-python backends are connected by a
                         thin grey line.
  Panel B (right):       per-method fit-runtime distribution on the same
                         log x. Horizontal bars (median + IQR) with
                         per-subject dots overlaid. Inline annotation:
                         AMICA NumPy-CPU / JAX-GPU ratio.

Inputs:
  --csv  long-form benchmark CSV with one row per (subject, method).
         Default: ../cc_benchmark/results/v3_paper_stage1_cluster/benchmark_results.csv

Outputs:
  --out  PDF (default: figures/fig_quality_cost.pdf)
  Companion fig_quality_cost_stats.csv with per-method mean MIR + 95% CI
  and median runtime + IQR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from style import (
    DISPLAY_INLINE, DISPLAY_STACKED, MARKERS, PALETTE, set_paper_style,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "cc_benchmark" / "results" / "v3_paper_stage1_cluster" / "benchmark_results.csv"
DEFAULT_OUT = HERE / "figures" / "fig_quality_cost.pdf"


def main(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df[["subject", "method", "mir_kbits_s", "fit_runtime_s"]].dropna()
    set_paper_style()

    # per-method summary
    rows = []
    for m, g in df.groupby("method"):
        mir_mean = float(g["mir_kbits_s"].mean())
        mir_sem  = float(g["mir_kbits_s"].std(ddof=1) / np.sqrt(len(g)))
        mir_ci95 = float(stats.t.ppf(0.975, len(g) - 1) * mir_sem)
        rt_med = float(g["fit_runtime_s"].median())
        rt_q1  = float(g["fit_runtime_s"].quantile(0.25))
        rt_q3  = float(g["fit_runtime_s"].quantile(0.75))
        rows.append({
            "method": m, "mir_mean": mir_mean, "mir_ci95": mir_ci95,
            "runtime_median": rt_med, "runtime_q1": rt_q1, "runtime_q3": rt_q3,
            "n": len(g),
        })
    S = pd.DataFrame(rows)

    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)

    # --- Panel A: scatter ----------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    amica_pts = S[S["method"].str.startswith("AMICA")].sort_values("runtime_median")
    axA.plot(
        amica_pts["runtime_median"], amica_pts["mir_mean"],
        color="0.55", linewidth=1.0, linestyle="-", zorder=1,
    )
    for _, r in S.iterrows():
        m = r["method"]
        rt = r["runtime_median"]
        mir = r["mir_mean"]
        axA.errorbar(
            rt, mir,
            xerr=[[rt - r["runtime_q1"]], [r["runtime_q3"] - rt]],
            yerr=r["mir_ci95"],
            fmt=MARKERS[m], color=PALETTE[m], ecolor=PALETTE[m],
            markersize=10, markeredgecolor="black", markeredgewidth=0.6,
            capsize=2.5, lw=1.0, alpha=0.95, zorder=3,
        )
        ha, va, dx, dy = "left", "bottom", 1.10, 1.012
        if m == "AMICA-Python (JAX-GPU)":
            ha, dx = "right", 0.93
        elif m == "Infomax":
            ha, dx, dy = "right", 0.92, 1.006
        elif m == "FastICA":
            ha, dx, dy = "left", 1.05, 0.978
        elif m == "Picard":
            ha, dx, dy = "left", 1.05, 0.978
        elif m == "AMICA-Python (JAX-CPU)":
            ha, dx, dy = "right", 0.95, 1.02
        elif m == "AMICA-Python (NumPy-CPU)":
            ha, dx, dy = "left", 1.04, 1.02
        axA.text(
            rt * dx, mir * dy, DISPLAY_INLINE[m],
            ha=ha, va=va, fontsize=8.5, color=PALETTE[m], fontweight="bold",
        )

    axA.set_xscale("log")
    axA.set_xlabel("Median fit runtime per subject (s, log scale)")
    axA.set_ylabel("MIR (kbits/s), mean " + r"$\pm$ 95% CI")
    axA.set_ylim(3.9, 5.1)
    axA.set_xlim(30, 60000)
    axA.annotate(
        "better", xy=(45, 5.05), xytext=(220, 5.05),
        ha="center", va="center", fontsize=9, color="0.35", fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0),
    )
    axA.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    axA.set_title("A. Quality–cost trade-off",
                  loc="left", fontsize=10, fontweight="bold")

    # --- Panel B: runtime distribution ---------------------------------
    axB = fig.add_subplot(gs[0, 1])
    S_sorted = S.sort_values("runtime_median").reset_index(drop=True)
    y_pos = np.arange(len(S_sorted))
    for i, r in S_sorted.iterrows():
        m = r["method"]
        g = df[df["method"] == m]
        rt = r["runtime_median"]
        axB.barh(
            i, rt, height=0.6,
            color=PALETTE[m], edgecolor="black", linewidth=0.4, alpha=0.65,
            xerr=[[rt - r["runtime_q1"]], [r["runtime_q3"] - rt]],
            error_kw=dict(ecolor="black", lw=0.6, capsize=2),
        )
        rng = np.random.default_rng(7 + i)
        jitter = rng.normal(0, 0.10, size=len(g))
        axB.scatter(
            g["fit_runtime_s"], np.full(len(g), i) + jitter,
            s=12, c="black", alpha=0.55, linewidths=0, zorder=5,
        )

    axB.set_xscale("log")
    axB.set_xlim(30, 100000)
    axB.set_yticks(y_pos)
    axB.set_yticklabels([DISPLAY_STACKED[m] for m in S_sorted["method"]],
                        fontsize=8.5)
    axB.set_xlabel("Fit runtime per subject (s, log scale)")
    axB.set_title("B. Per-subject runtime distribution",
                  loc="left", fontsize=10, fontweight="bold")
    axB.grid(axis="x", which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    gpu_med = float(
        S_sorted[S_sorted["method"] == "AMICA-Python (JAX-GPU)"]["runtime_median"].iloc[0]
    )
    numpy_med = float(
        S_sorted[S_sorted["method"] == "AMICA-Python (NumPy-CPU)"]["runtime_median"].iloc[0]
    )
    ratio = numpy_med / gpu_med
    axB.text(
        0.97, 0.06,
        f"AMICA NumPy-CPU / JAX-GPU\n$\\approx {ratio:.0f}\\times$ slower",
        transform=axB.transAxes, ha="right", va="bottom",
        fontsize=8.5, color="#08306b", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#08306b",
                  lw=0.6, alpha=0.9),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    stats_csv = out_path.with_name(out_path.stem + "_stats.csv")
    S.to_csv(stats_csv, index=False)
    print(f"WROTE {out_path}  ({out_path.stat().st_size} B)")
    print(f"WROTE {stats_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"benchmark_results.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output PDF path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    main(args.csv, args.out)
