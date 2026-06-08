"""
Render the zenodo preprint MIR figure (main Fig 4).

Two-panel composite:
  Panel A (left, narrow): across-subject mean MIR per method, sorted high
                          to low, 95% CI whiskers (bar chart).
  Panel B (right, wide):  per-subject paired DeltaMIR (AMICA - comparator)
                          for the three contrasts, with Cohen's d_z and
                          Holm-adjusted paired t-test p annotated inline.

Inputs:
  --csv  long-form benchmark CSV with one row per (subject, method).
         Default: ../cc_benchmark/results/v3_paper_stage1_cluster/benchmark_results.csv

Outputs:
  --out  PDF (default: figures/fig_mir_combined.pdf)
  Companion fig_mir_combined_stats.csv with the three contrast statistics
  (mean delta, sd, 95% CI, d_z, t, p_raw, p_holm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from style import DISPLAY_INLINE, PALETTE, set_paper_style

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "cc_benchmark" / "results" / "v3_paper_stage1_cluster" / "benchmark_results.csv"
DEFAULT_OUT = HERE / "figures" / "fig_mir_combined.pdf"


def _holm(p_raws: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as input."""
    order = sorted(range(len(p_raws)), key=lambda i: p_raws[i])
    adj = [0.0] * len(p_raws)
    k = len(p_raws)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p_raws[idx] * (k - rank))
    return adj


def main(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)[["subject", "method", "mir_kbits_s"]].dropna()
    set_paper_style()

    # ----- Panel A: across-subject mean MIR per method, sorted ----------
    summary = (
        df.groupby("method")["mir_kbits_s"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = summary["sem"] * stats.t.ppf(0.975, summary["count"] - 1)
    summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)

    # ----- Panel B: paired DeltaMIR (AMICA collapsed across backends) ---
    amica_methods = [m for m in df["method"].unique() if m.startswith("AMICA")]
    amica_pivot = (
        df[df["method"].isin(amica_methods)]
        .pivot_table(index="subject", columns="method", values="mir_kbits_s")
        .mean(axis=1)
        .rename("AMICA")
    )

    contrasts = []
    p_raws = []
    for comp in ["Picard", "Infomax", "FastICA"]:
        comp_series = (
            df[df["method"] == comp].set_index("subject")["mir_kbits_s"].rename(comp)
        )
        paired = pd.concat([amica_pivot, comp_series], axis=1).dropna()
        delta = paired["AMICA"] - paired[comp]
        n = len(delta)
        mean_d = float(delta.mean())
        sd_d = float(delta.std(ddof=1))
        sem_d = sd_d / np.sqrt(n)
        ci95 = float(stats.t.ppf(0.975, n - 1) * sem_d)
        d_z = mean_d / sd_d
        t_stat, p_ttest = stats.ttest_rel(paired["AMICA"], paired[comp])
        contrasts.append({
            "comparator": comp, "n": n, "mean_delta": mean_d, "sd_delta": sd_d,
            "ci95": ci95, "d_z": d_z, "t_stat": float(t_stat),
            "p_raw": float(p_ttest), "deltas": delta.values.tolist(),
        })
        p_raws.append(float(p_ttest))
    for c, p_holm in zip(contrasts, _holm(p_raws)):
        c["p_holm"] = p_holm

    # ----- plotting -----------------------------------------------------
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.7], wspace=0.45)

    # --- Panel A ---
    axA = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(summary))[::-1]
    colors = [PALETTE[m] for m in summary["method"]]
    labels = [DISPLAY_INLINE[m] for m in summary["method"]]
    axA.barh(
        y_pos, summary["mean"], xerr=summary["ci95"],
        color=colors, edgecolor="black", linewidth=0.4,
        error_kw=dict(ecolor="black", lw=0.7, capsize=2), alpha=0.9,
    )
    axA.set_yticks(y_pos)
    axA.set_yticklabels(labels, fontsize=8)
    axA.set_xlabel("MIR (kbits/s), mean " + r"$\pm$ 95% CI")
    axA.set_title("A. Across-subject mean MIR",
                  loc="left", fontsize=10, fontweight="bold")
    axA.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    axA.set_xlim(0, max(summary["mean"]) * 1.18)

    # --- Panel B ---
    axB = fig.add_subplot(gs[0, 1])
    y_rows = np.arange(len(contrasts))[::-1]
    for i, c in enumerate(contrasts):
        y = y_rows[i]
        rng = np.random.default_rng(i * 7 + 1)
        jitter = rng.normal(0, 0.06, size=len(c["deltas"]))
        axB.scatter(
            c["deltas"], np.full_like(c["deltas"], y, dtype=float) + jitter,
            s=14, c="0.55", alpha=0.7, linewidths=0,
        )
        axB.errorbar(
            c["mean_delta"], y, xerr=c["ci95"],
            fmt="s", color="black", markersize=8,
            capsize=4, lw=1.4, zorder=5,
        )
        p_str = (
            f"$p_{{\\mathrm{{Holm}}}}<10^{{{int(np.floor(np.log10(c['p_holm'])))}}}$"
            if c["p_holm"] < 1e-3 else f"$p_{{\\mathrm{{Holm}}}}={c['p_holm']:.3g}$"
        )
        axB.text(
            0, y, f"$d_z = {c['d_z']:.2f}$ ; {p_str}",
            ha="left", va="center", fontsize=8, transform=axB.transData,
        )

    axB.axvline(0, color="black", linewidth=0.7, linestyle="--")
    axB.set_yticks(y_rows)
    axB.set_yticklabels([f"AMICA $-$ {c['comparator']}" for c in contrasts])
    axB.set_xlabel(r"$\Delta$MIR (kbits/s)  $=$  MIR$_{\mathrm{AMICA}} - $ MIR$_{\mathrm{comparator}}$")
    axB.set_title("B. Paired per-subject $\\Delta$MIR (n = 25)",
                  loc="left", fontsize=10, fontweight="bold")
    axB.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    lo, hi = axB.get_xlim()
    axB.set_xlim(lo, hi + (hi - lo) * 0.35)
    for txt, _, y in zip(axB.texts, contrasts, y_rows):
        txt.set_position((hi + (hi - lo) * 0.02, y))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    stats_csv = out_path.with_name(out_path.stem + "_stats.csv")
    pd.DataFrame([
        {k: v for k, v in c.items() if k != "deltas"} for c in contrasts
    ]).to_csv(stats_csv, index=False)
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
