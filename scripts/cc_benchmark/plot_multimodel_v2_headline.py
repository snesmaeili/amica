"""Part C v2 headline figure: real cohorts vs TWO stationary nulls on one axis.

The decisive contrast the original figures never co-located. Panel A (the headline):
delta-LL(H) = LL(H)-LL(1) -- the genuine non-stationarity signal -- for the four real
cohorts, well above a shaded band spanned by the within-data phase-randomized surrogate
null and the synthetic stationary null. Panel B (the caution): N_eff(H), where the
phase surrogate sits ABOVE the real cohorts (it tracks Gaussianity / model count, not
non-stationarity) while the non-Gaussian synthetic stationary control stays low -- so
N_eff is not a valid stationarity signal.

Inputs: results/multimodel_metrics_v2/<cohort>/metrics_aggregate_H.csv  +
        results/multimodel_synthetic_2000/synthetic_summary.json
Run:    python scripts/cc_benchmark/plot_multimodel_v2_headline.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[2]  # repo root (repos/amica-mm)
MET = HERE / "results" / "multimodel_metrics_v2"
SYN = HERE / "results" / "multimodel_synthetic_2000" / "synthetic_summary.json"
OUT = HERE / "results" / "multimodel_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

REAL = [  # dir, label, color, marker
    ("ds004505",      "task, 64 ch",  "#B2182B", "o"),
    ("ds004505_ch19", "task, 19 ch (matched)", "#E08214", "s"),
    ("ds004504",      "rest, 19 ch",  "#4393C3", "^"),
    ("ds004621",      "rest, 128 ch", "#2166AC", "D"),
]
SURR = ["ds004505_surr", "ds004504_surr", "ds004621_surr"]


def agg(d):
    p = MET / d / "metrics_aggregate_H.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    H = np.array([int(r["H"]) for r in rows])
    out = {"H": H}
    for k in ("delta_ll_mean", "delta_ll_sem", "n_eff_mean", "n_eff_sem"):
        out[k] = np.array([float(r[k]) for r in rows])
    return out


def syn_curve():
    d = json.load(SYN.open())["stationary"]
    H = np.array([r["H"] for r in d])
    ll = np.array([r["ll"] for r in d])
    return H, ll - ll[0], np.array([r["n_eff"] for r in d])


real = {d: agg(d) for d, *_ in REAL}
surr = {d: agg(d) for d in SURR}
sH, sdLL, sNeff = syn_curve()

# null band = envelope across the 3 phase surrogates (per H)
Hs = real["ds004505"]["H"]
def band(metric):
    M = np.vstack([surr[d][metric] for d in SURR if surr[d] is not None])
    return M.min(0), M.max(0)
dll_lo, dll_hi = band("delta_ll_mean")
neff_lo, neff_hi = band("n_eff_mean")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3))

# ---- Panel A: delta-LL (the signal) ----
axA.fill_between(Hs, dll_lo, dll_hi, color="0.6", alpha=0.28, lw=0,
                 label="phase-surrogate null (3 cohorts)")
axA.plot(sH, sdLL, color="0.35", ls=":", lw=1.6, label="synthetic stationary null")
for d, lab, c, m in REAL:
    r = real[d]
    axA.errorbar(r["H"], r["delta_ll_mean"], yerr=r["delta_ll_sem"], color=c, marker=m,
                 ms=5, lw=1.8, capsize=2, label=lab)
axA.set_xlabel("number of models $H$")
axA.set_ylabel(r"$\Delta\mathrm{LL}(H)=\mathrm{LL}(H)-\mathrm{LL}(1)$  (nats/sample)")
axA.set_title("A. Likelihood gain — the non-stationarity signal", loc="left", fontsize=10, fontweight="bold")
axA.set_xticks(range(1, 11))
axA.set_ylim(-0.004, None)
axA.axhline(0, color="0.8", lw=0.8, zorder=0)
axA.grid(True, ls=":", lw=0.5, alpha=0.5)
axA.annotate("real EEG $\\gg$ stationary nulls\n(7–316$\\times$ at $H{=}10$)", xy=(6.0, 0.052),
             fontsize=8.6, color="0.25", fontstyle="italic", ha="center")
axA.legend(loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95, edgecolor="0.85",
           ncol=1, handlelength=1.4, labelspacing=0.25)

# ---- Panel B: N_eff (the caution) ----
axB.fill_between(Hs, neff_lo, neff_hi, color="0.6", alpha=0.28, lw=0,
                 label="phase-surrogate null")
axB.plot(sH, sNeff, color="0.35", ls=":", lw=1.6, label="synthetic stationary null")
for d, lab, c, m in REAL:
    r = real[d]
    axB.plot(r["H"], r["n_eff_mean"], color=c, marker=m, ms=5, lw=1.8, label=lab)
axB.plot([1, 10], [1, 10], color="0.8", ls="--", lw=0.9, zorder=0)
axB.set_xlabel("number of models $H$")
axB.set_ylabel(r"$N_{\mathrm{eff}}=\exp(-\sum_h \pi_h\log\pi_h)$")
axB.set_title("B. Effective #models — a confounded metric", loc="left", fontsize=10, fontweight="bold")
axB.set_xticks(range(1, 11))
axB.grid(True, ls=":", lw=0.5, alpha=0.5)
axB.annotate("the Gaussian surrogate inflates $N_{\\mathrm{eff}}$\n$above$ real EEG → not a stationarity signal",
             xy=(5.4, 4.0), fontsize=8.4, color="0.25", fontstyle="italic", ha="center")
axB.legend(loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95, edgecolor="0.85",
           handlelength=1.4, labelspacing=0.25)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_mm_headline_deltaLL.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "fig_mm_headline_deltaLL.pdf")
# quick numeric trace for the caption
for d, lab, *_ in REAL:
    r = real[d]
    print(f"  {lab:<22} dLL@10={r['delta_ll_mean'][-1]:.4f}  N_eff@10={r['n_eff_mean'][-1]:.2f}")
print(f"  surrogate dLL@10 band: {dll_lo[-1]:.4f}-{dll_hi[-1]:.4f}  N_eff band: {neff_lo[-1]:.2f}-{neff_hi[-1]:.2f}")
print(f"  synthetic dLL@10={sdLL[-1]:.4f}  N_eff@10={sNeff[-1]:.2f}")
