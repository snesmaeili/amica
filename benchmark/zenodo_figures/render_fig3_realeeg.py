"""Main Figure 3 — replicated real-EEG MIR advantage with visible PMI + dipolarity limits.

Central scientific result: amica-python has a positive within-subject MIR advantage over all
three comparators in all three datasets, accompanied by slightly higher remnant PMI on
ds004505 and no method separation in dipolarity. Panels (guideline §10):
  (a) subject-level MIR differences (9 dataset x comparator rows; dot/subject + mean ± 95% CI)
  (b) cross-dataset standardized effects (d_z forest)
  (c) remnant-PMI secondary result (positive = amica worse)
  (d) dipolarity secondary result (near-dipolar fraction, no detectable separation)

Reads the committed subject-level CSVs + estimates_realeeg.csv (Stage 2). No new fits.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "cc_benchmark" / "results"
CSV = {
    "ds004505": RESULTS / "v3_paper_stage1_cluster" / "benchmark_results.csv",
    "ds004504": RESULTS / "ds004504_v3" / "benchmark_results.csv",
    "ds004621": RESULTS / "ds004621_v3" / "benchmark_results.csv",
}
AMICA = "AMICA-Python (JAX-GPU)"
COMPS = ["Picard", "Infomax", "FastICA"]
CCOL = {"Picard": style.PICARD, "Infomax": style.INFOMAX, "FastICA": style.FASTICA}
CLAB = {"Picard": "Picard", "Infomax": "Ext. Infomax", "FastICA": "FastICA"}
DS_LABEL = {"ds004505": "ds004505\ntask 64ch", "ds004504": "ds004504\nrest 19ch",
            "ds004621": "ds004621\nrest 64ch"}


def paired_diff(df, metric, comp):
    a = df[df.method == AMICA].set_index("subject")[metric]
    c = df[df.method == comp].set_index("subject")[metric]
    j = pd.concat([a.rename("a"), c.rename("c")], axis=1).dropna()
    return (j["a"] - j["c"]).to_numpy()


def main():
    style.set_paper_style()
    data = {ds: pd.read_csv(p) for ds, p in CSV.items()}
    est = pd.read_csv(HERE / "estimates_realeeg.csv")
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.34)
    ax_b = fig.add_subplot(gs[0, 0])   # a: subject MIR diffs (dominant)
    ax_c = fig.add_subplot(gs[0, 1])   # b: d_z forest
    ax_d = fig.add_subplot(gs[1, 0])   # c: remnant PMI
    ax_e = fig.add_subplot(gs[1, 1])   # d: dipolarity

    # row layout: 9 rows grouped by dataset (3 comparators each), top-to-bottom
    rows = [(ds, c) for ds in CSV for c in COMPS]
    ypos = {rc: len(rows) - 1 - i for i, rc in enumerate(rows)}

    # ---- panel a: subject-level MIR differences ----
    for (ds, comp), y in ypos.items():
        d = paired_diff(data[ds], "mir_kbits_s", comp)
        jit = (rng.random(len(d)) - 0.5) * 0.55
        ax_b.scatter(d, np.full(len(d), y) + jit, s=8, color=CCOL[comp], alpha=0.45,
                     edgecolors="none", zorder=2)
        m, lo, hi = style.bootstrap_ci(d)
        ax_b.plot([lo, hi], [y, y], color="k", lw=1.4, zorder=4)
        ax_b.plot(m, y, "s", ms=5, color=CCOL[comp], mec="k", mew=0.6, zorder=5)
    ax_b.axvline(0, color="#888", lw=0.8, zorder=1)
    ax_b.set_yticks(list(ypos.values()))
    ax_b.set_yticklabels([CLAB[c] for ds, c in ypos])
    ax_b.set_ylim(-0.7, len(rows) - 0.3)
    ax_b.set_xlabel("$\\Delta$MIR = amica $-$ comparator  (kbit/s)")
    ax_b.set_title("a  Subject-level MIR advantage", loc="left", fontweight="bold", fontsize=9.5)
    # dataset group labels (left of panel a) + faint separators on every panel
    for ds in CSV:
        ys = [ypos[(ds, c)] for c in COMPS]
        ax_b.text(-0.52, np.mean(ys), ds, transform=ax_b.get_yaxis_transform(),
                  rotation=90, va="center", ha="center", fontsize=7.5, color="#333",
                  fontweight="semibold")
    for ax in (ax_b, ax_c, ax_d, ax_e):
        for ysep in (5.5, 2.5):
            ax.axhline(ysep, color="#e3e3e3", lw=0.6, zorder=0)

    # ---- panel b: cross-dataset d_z forest ----
    mir = est[est.metric == "mir_kbits_s"].set_index(["dataset", "comparator"])
    for (ds, comp), y in ypos.items():
        r = mir.loc[(ds, comp)]
        ax_c.plot([r.dz_lo, r.dz_hi], [y, y], color="k", lw=1.2, zorder=3)
        ax_c.plot(r.dz, y, "o", ms=5, color=CCOL[comp], mec="k", mew=0.6, zorder=4)
        ax_c.annotate(f"{int(r.favours_amica)}/{int(r.n)}", (r.dz_hi, y), xytext=(4, 0),
                      textcoords="offset points", va="center", fontsize=6.5, color="#444")
    ax_c.axvline(0, color="#888", lw=0.8)
    ax_c.set_yticks(list(ypos.values())); ax_c.set_yticklabels([])
    ax_c.set_ylim(-0.7, len(rows) - 0.3)
    ax_c.set_xlabel("Cohen's $d_z$ (95% CI)")
    ax_c.set_title("b  Standardized effect", loc="left", fontweight="bold", fontsize=9.5)

    # ---- panel c: remnant PMI (positive = amica worse) ----
    pmi = est[est.metric == "remnant_pmi_percent"].set_index(["dataset", "comparator"])
    for (ds, comp), y in ypos.items():
        r = pmi.loc[(ds, comp)]
        ax_d.plot([r.ci_lo, r.ci_hi], [y, y], color="k", lw=1.2)
        ax_d.plot(r.mean_diff, y, "o", ms=5, color=CCOL[comp], mec="k", mew=0.6)
    ax_d.axvline(0, color="#888", lw=0.8)
    ax_d.set_yticks(list(ypos.values())); ax_d.set_yticklabels([CLAB[c] for ds, c in ypos])
    ax_d.set_ylim(-0.7, len(rows) - 0.3)
    ax_d.set_xlabel("$\\Delta$ remnant PMI (pp)  →  amica worse")
    ax_d.set_title("c  Residual dependence (secondary)", loc="left", fontweight="bold", fontsize=9.5)

    # ---- panel d: dipolarity nd@5 + nd@10 (no detectable separation) ----
    nd5 = est[est.metric == "nd_5_percent"].set_index(["dataset", "comparator"])
    nd10 = est[est.metric == "nd_10_percent"].set_index(["dataset", "comparator"])
    for (ds, comp), y in ypos.items():
        r5, r10 = nd5.loc[(ds, comp)], nd10.loc[(ds, comp)]
        ax_e.plot([r5.ci_lo, r5.ci_hi], [y + 0.12, y + 0.12], color=CCOL[comp], lw=1.1, alpha=0.9)
        ax_e.plot(r5.mean_diff, y + 0.12, "o", ms=4, color=CCOL[comp], mec="none")
        ax_e.plot([r10.ci_lo, r10.ci_hi], [y - 0.12, y - 0.12], color=CCOL[comp], lw=1.1, alpha=0.4)
        ax_e.plot(r10.mean_diff, y - 0.12, "D", ms=3.5, color=CCOL[comp], mec="none", alpha=0.6)
    ax_e.axvline(0, color="#888", lw=0.8)
    ax_e.set_yticks(list(ypos.values())); ax_e.set_yticklabels([])
    ax_e.set_ylim(-0.7, len(rows) - 0.3)
    ax_e.set_xlabel("$\\Delta$ near-dipolar fraction (pp)\n●  RV<5%    ◆  RV<10%")
    ax_e.set_title("d  Dipolarity (secondary)", loc="left", fontweight="bold", fontsize=9.5)

    fig.suptitle("amica-python: replicated MIR advantage with secondary PMI / dipolarity limits "
                 "(3 datasets, 96 subjects)", fontsize=10, fontweight="bold", y=0.995)
    out = style.save_vector(fig, HERE / "out" / "fig3_realeeg_benchmark.pdf")
    print(f"wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
