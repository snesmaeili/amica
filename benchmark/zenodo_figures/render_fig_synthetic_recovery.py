"""
Render the zenodo preprint synthetic-recovery figure (main Fig 1).

Layout: 2 rows (Laplacian / Mixture) x 3 columns (|r|_topo / |r|_source /
Amari distance), grouped bars by method, per-seed dots overlaid.

Inputs:
  --csv   tidy long-form CSV with columns
          {regime, condition, method_label, seed, gt_r_topo_median,
           gt_r_source_median, gt_amari_index, ...}
          Default location:
          ../mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv

Outputs:
  --out   PDF (default: figures/fig_synthetic_recovery.pdf)
  Companion fig_synthetic_recovery_stats.csv with the median + IQR
  per (regime, condition, method, metric) so the visual can be re-derived.

Usage:
  python render_fig_synthetic_recovery.py \
      --csv  ../mne_synthetic/results/v1_full_analysis/synthetic_long_all_metrics.csv \
      --out  /path/to/figures/fig_synthetic_recovery.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import METHODS_SYNTH, PALETTE, set_paper_style

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "mne_synthetic" / "results" / "v1_full_analysis" / "synthetic_long_all_metrics.csv"
DEFAULT_OUT = HERE / "figures" / "fig_synthetic_recovery.pdf"

CONDITIONS = ["clean", "noise", "noise_eog"]
COND_LABELS = ["clean", "noise", "noise+EOG"]
REGIMES = ["Laplacian", "Mixture"]
METRICS = [
    ("gt_r_topo_median",   r"$|r|_{\mathrm{topo}}$"),
    ("gt_r_source_median", r"$|r|_{\mathrm{source}}$"),
    ("gt_amari_index",     "Amari distance"),
]


def main(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    set_paper_style()

    fig, axes = plt.subplots(
        len(REGIMES), len(METRICS),
        figsize=(9.5, 5.4),
        sharex=False,
    )

    bar_w = 0.16
    x_centres = np.arange(len(CONDITIONS))
    stats_rows = []

    for row_i, regime in enumerate(REGIMES):
        for col_i, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            for m_i, m in enumerate(METHODS_SYNTH):
                offset = (m_i - (len(METHODS_SYNTH) - 1) / 2.0) * bar_w
                medians, iqr_lo, iqr_hi = [], [], []
                scatter_x, scatter_y = [], []
                for c_i, cond in enumerate(CONDITIONS):
                    sub = df[
                        (df["regime"] == regime)
                        & (df["condition"] == cond)
                        & (df["method_label"] == m)
                    ]
                    vals = sub[metric_key].dropna().to_numpy()
                    if len(vals) == 0:
                        medians.append(np.nan)
                        iqr_lo.append(np.nan)
                        iqr_hi.append(np.nan)
                        continue
                    med = float(np.median(vals))
                    q1, q3 = np.quantile(vals, [0.25, 0.75])
                    medians.append(med)
                    iqr_lo.append(med - q1)
                    iqr_hi.append(q3 - med)
                    stats_rows.append({
                        "regime": regime, "condition": cond, "method": m,
                        "metric": metric_key, "median": med,
                        "q1": float(q1), "q3": float(q3),
                        "n_seeds": int(len(vals)),
                    })
                    jitter = np.random.default_rng(c_i * 100 + m_i).normal(
                        0, bar_w * 0.12, size=len(vals)
                    )
                    scatter_x.extend(x_centres[c_i] + offset + jitter)
                    scatter_y.extend(vals.tolist())
                ax.bar(
                    x_centres + offset,
                    medians,
                    width=bar_w,
                    color=PALETTE[m],
                    edgecolor="black",
                    linewidth=0.4,
                    label=m if (row_i == 0 and col_i == 0) else None,
                    yerr=[iqr_lo, iqr_hi],
                    error_kw=dict(ecolor="black", lw=0.6, capsize=2),
                    alpha=0.9,
                )
                ax.scatter(
                    scatter_x, scatter_y,
                    s=4, c="black", alpha=0.4, linewidths=0,
                )
            ax.set_xticks(x_centres)
            ax.set_xticklabels(COND_LABELS)
            if metric_key.startswith("gt_r_"):
                ax.set_ylim(0, 1.02)
            if col_i == 0:
                ax.set_ylabel(f"{regime}\n{metric_label}",
                              fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel(metric_label)
            if row_i == 0:
                ax.set_title(metric_label)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(METHODS_SYNTH),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    stats_csv = out_path.with_name(out_path.stem + "_stats.csv")
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    print(f"WROTE {out_path}  ({out_path.stat().st_size} B)")
    print(f"WROTE {stats_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Tidy long-form CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output PDF path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    main(args.csv, args.out)
