"""Main Figure 5 — the 3,000-iteration benchmark stops before convergence.

All reported ds004505 AMICA decompositions end at a fixed computational budget while the
likelihood is still improving; the benchmark compares methods at specified budgets, not at
matched optima. Panels (guideline §10):
  (a) relative objective improvement LL_i - LL_0 (faint subject traces + median + IQR)
  (b) per-iteration increment (rolling median positive dLL, log) vs a stopping tolerance
  (c) residual end-of-run slope (median dLL over iters 2,751-3,000, per subject)

Reads the committed ds004505 iteration trace (JAX-GPU, 25 subjects). No new fits.
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
TRACE = (HERE.parent / "cc_benchmark" / "results" / "v3_paper_stage1_cluster"
         / "iteration_trace.csv.gz")
AMICA = "AMICA-Python (JAX-GPU)"
NEWT_START = 50            # config newt_start (quasi-Newton onset)
TOL = 1e-9                 # AMICA default min_dll convergence tolerance
MILESTONES = (250, 1000, 2000)


def main():
    style.set_paper_style()
    df = pd.read_csv(TRACE)
    df = df[df.method == AMICA][["subject", "iteration", "log_likelihood"]].dropna()
    subs = sorted(df.subject.unique())
    # pivot: iteration x subject  (LL)
    piv = df.pivot_table(index="iteration", columns="subject", values="log_likelihood").sort_index()
    iters = piv.index.to_numpy()

    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    blue = style.AMICA_BLUE

    # ---- a: relative LL improvement (LL_i - LL_0) ----
    rel = piv - piv.iloc[0]
    axs[0].plot(iters, rel.to_numpy(), color=blue, lw=0.35, alpha=0.25, zorder=1)
    med = rel.median(axis=1)
    q1, q3 = rel.quantile(0.25, axis=1), rel.quantile(0.75, axis=1)
    axs[0].fill_between(iters, q1, q3, color=blue, alpha=0.20, zorder=2)
    axs[0].plot(iters, med, color=blue, lw=1.6, zorder=3)
    axs[0].axvline(NEWT_START, color="#888", lw=0.8, ls="-")
    axs[0].text(NEWT_START + 40, axs[0].get_ylim()[0], " Newton on", fontsize=6.5, color="#555",
                rotation=90, va="bottom")
    for ms in MILESTONES:
        axs[0].axvline(ms, color="#bbb", lw=0.7, ls=":", zorder=0)
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("LL$_i$ $-$ LL$_0$ (nats/sample)")
    axs[0].set_title("a  Relative improvement", loc="left", fontweight="bold", fontsize=9.5)

    # ---- b: per-iteration increment dLL (rolling median), log ----
    dll = piv.diff()
    dll_pos = dll.clip(lower=1e-12)
    roll = dll_pos.rolling(25, min_periods=1).median()
    rmed = roll.median(axis=1)
    rq1, rq3 = roll.quantile(0.10, axis=1), roll.quantile(0.90, axis=1)
    axs[1].fill_between(iters, rq1, rq3, color=blue, alpha=0.18)
    axs[1].plot(iters, rmed, color=blue, lw=1.5)
    axs[1].axhline(TOL, color="#D55E00", lw=1.0, ls="--")
    axs[1].text(iters[-1], TOL * 1.3, f" tol $\\approx${TOL:g}", fontsize=6.5, color="#D55E00",
                ha="right", va="bottom")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("rolling median $\\Delta$LL")
    axs[1].set_title("b  Increment per iteration", loc="left", fontweight="bold", fontsize=9.5)
    tail_med = float(dll[(dll.index >= 2751)].median(axis=0).median())
    axs[1].text(0.96, 0.93, f"near-plateau;\nmedian residual {tail_med:.0e}\n($\\approx${tail_med/TOL:.0f}$\\times$ tol)",
                transform=axs[1].transAxes, ha="right", va="top", fontsize=6.6, color="#555")

    # ---- c: residual end-of-run slope (median dLL over 2,751-3,000) ----
    tail = dll[(dll.index >= 2751)].median(axis=0).clip(lower=1e-12)
    yj = (np.random.default_rng(0).random(len(tail)) - 0.5) * 0.5
    axs[2].scatter(tail.to_numpy(), yj, s=14, color=blue, alpha=0.6, edgecolors="none")
    m, lo, hi = style.bootstrap_ci(tail.to_numpy(), statistic=np.median)
    axs[2].plot([lo, hi], [0, 0], color="k", lw=1.4)
    axs[2].plot(m, 0, "s", ms=6, color=blue, mec="k", mew=0.6)
    axs[2].axvline(TOL, color="#D55E00", lw=1.0, ls="--")
    axs[2].set_xscale("log"); axs[2].set_yticks([]); axs[2].set_ylim(-0.6, 0.6)
    axs[2].set_xlabel("median $\\Delta$LL, iters 2,751–3,000")
    axs[2].set_title("c  Residual end-of-run slope", loc="left", fontweight="bold", fontsize=9.5)

    fig.suptitle("ds004505 AMICA stops at a fixed 3,000-iteration budget; the likelihood is "
                 f"near-plateau ($N={len(subs)}$)", fontsize=9.6, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = style.save_vector(fig, HERE / "out" / "fig5_convergence.pdf")
    print(f"wrote {out}; median tail dLL={m:.2e} (~{m/TOL:.0f}x the {TOL:g} tol)")


if __name__ == "__main__":
    main()
