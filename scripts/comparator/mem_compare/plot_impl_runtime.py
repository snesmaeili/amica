"""Cross-implementation runtime/speed figure (companion to fig_mem_comparison.pdf).

Same five AMICA implementations as the memory comparison -- amica-python (JAX, full-batch + chunked),
scott-huberty (PyTorch), pyamica (PyTorch), Fortran AMICA 1.7 -- on ds004505, CPU, at a matched low
iteration budget (100 iter sub-01 / 60 iter sub-02..06). Two panels, mirroring the method-comparison
fig_quality_cost but for IMPLEMENTATIONS:

  Panel A: final log-likelihood (the AMICA objective; ~identical across implementations -> they reach
           the same optimum) vs per-iteration fit runtime (log x).
  Panel B: per-implementation per-iteration runtime distribution (median + IQR, per-subject dots).

Runtime is reported PER ITERATION (fit_time_s / n_iter) so the 100-iter and 60-iter runs are
comparable. GPU is intentionally excluded: the 100-iter GPU fit time is dominated by JAX's one-time
JIT compile, which would misrepresent steady-state per-iteration speed (see PROVENANCE.md).

With --json-root PATH (the workspace results dir holding mem_compare/ + mem_multisubj/) this rebuilds
impl_runtime_table.csv from the result JSONs; otherwise it plots the committed CSV.
Run: python plot_impl_runtime.py --json-root D:/amica-validation-workspace/results
     python plot_impl_runtime.py            # from committed CSV
"""
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[sys.argv.index("--json-root") + 1]) if "--json-root" in sys.argv else None

# Implementation labels/colours -- shared with the sibling memory figure (plot_mem_comparison.py).
LBL_STACK = {
    "amica_python_jax": "AMICA-Python\n(full-batch)",
    "amica_python_jax_chunked": "AMICA-Python\n(chunked)",
    "scott_huberty_torch": "scott-huberty\n(PyTorch)",
    "pyamica_torch": "pyamica\n(PyTorch)",
    "fortran_amica17": "Fortran\nAMICA 1.7",
}
LBL_INLINE = {k: v.replace("\n", " ") for k, v in LBL_STACK.items()}
COL = {
    "amica_python_jax": "#1F4E79", "amica_python_jax_chunked": "#5B9BD5",
    "scott_huberty_torch": "#D77A00", "pyamica_torch": "#2A9D8F", "fortran_amica17": "#7030A0",
}
ORDER = ["amica_python_jax", "amica_python_jax_chunked", "scott_huberty_torch",
         "pyamica_torch", "fortran_amica17"]


def _build(root):
    rows = []
    dirs = [root / "mem_compare/cpu/ds004505_sub-01_mem"]
    dirs += [Path(p) for p in sorted(glob.glob(str(root / "mem_multisubj/ds004505_sub-*_mem")))]
    for d in dirs:
        sub = Path(d).name.split("_")[1]
        for jf in glob.glob(str(Path(d) / "*.json")):
            a = json.load(open(jf))
            impl, ft, ni = a.get("implementation"), a.get("fit_time_s"), a.get("n_iter")
            if impl in LBL_STACK and ft and ni and a.get("device", "cpu") == "cpu":
                rows.append(dict(subject=sub, implementation=impl, ll_final=a.get("ll_final"),
                                 fit_time_s=ft, n_iter=ni, s_per_iter=ft / ni))
    rows.sort(key=lambda r: (ORDER.index(r["implementation"]), r["subject"]))
    with (HERE / "impl_runtime_table.csv").open("w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "implementation", "ll_final",
                                          "fit_time_s", "n_iter", "s_per_iter"])
        w.writeheader(); w.writerows(rows)
    return rows


def _load_csv():
    rows = list(csv.DictReader((HERE / "impl_runtime_table.csv").open(encoding="utf-8")))
    for r in rows:
        for k in ("ll_final", "fit_time_s", "s_per_iter"):
            r[k] = float(r[k])
        r["n_iter"] = int(r["n_iter"])
    return rows


rows = _build(ROOT) if ROOT else _load_csv()

# per-implementation summary
impls = [m for m in ORDER if any(r["implementation"] == m for r in rows)]
summ = {}
for m in impls:
    spi = np.array([r["s_per_iter"] for r in rows if r["implementation"] == m])
    ll = np.array([r["ll_final"] for r in rows if r["implementation"] == m])
    summ[m] = dict(med=np.median(spi), q1=np.percentile(spi, 25), q3=np.percentile(spi, 75),
                   ll=float(np.median(ll)), n=len(spi), spi=spi)

fig = plt.figure(figsize=(12.5, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.30)

# --- Panel A: log-likelihood vs runtime (reference subject sub-01; ll is only comparable
#     WITHIN a subject -- it is a per-sample likelihood that varies across recordings) ---
axA = fig.add_subplot(gs[0, 0])
markers = {"amica_python_jax": "o", "amica_python_jax_chunked": "o",
           "scott_huberty_torch": "s", "pyamica_torch": "D", "fortran_amica17": "^"}
sub01 = {r["implementation"]: r for r in rows if r["subject"] == "sub-01"}
for m in impls:
    if m not in sub01:
        continue
    r = sub01[m]
    axA.scatter(r["s_per_iter"], r["ll_final"], marker=markers[m], color=COL[m],
                s=140, edgecolor="black", linewidths=0.6, alpha=0.95, zorder=3,
                label=LBL_INLINE[m])
ll01 = [sub01[m]["ll_final"] for m in impls if m in sub01]
rt01 = [sub01[m]["s_per_iter"] for m in impls if m in sub01]
axA.set_xscale("log")
axA.set_xlabel("Fit runtime per iteration (s, log scale)")
axA.set_ylabel("Final log-likelihood (AMICA objective)")
axA.set_ylim(float(np.mean(ll01)) - 0.04, float(np.mean(ll01)) + 0.04)
axA.annotate("all five implementations reach the\nsame optimum ($\\Delta$ll $<10^{-3}$)",
             xy=(0.5, 0.85), xycoords="axes fraction", ha="center", fontsize=8.5,
             color="0.35", fontstyle="italic")
axA.annotate("faster", xy=(min(rt01) * 0.78, np.mean(ll01) - 0.030),
             xytext=(min(rt01) * 2.2, np.mean(ll01) - 0.030),
             ha="center", va="center", fontsize=9, color="0.35", fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0))
axA.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
axA.set_title("A. Quality–cost (ds004505 sub-01): same optimum, different speed", loc="left",
              fontsize=9.5, fontweight="bold")
axA.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.8",
           fontsize=8.5, handlelength=1.2, borderpad=0.4, labelspacing=0.3)

# --- Panel B: per-implementation runtime distribution ---
axB = fig.add_subplot(gs[0, 1])
order_b = sorted(impls, key=lambda m: summ[m]["med"])  # fastest at bottom
for i, m in enumerate(order_b):
    s = summ[m]
    axB.barh(i, s["med"], height=0.6, color=COL[m], edgecolor="black", linewidth=0.4, alpha=0.65,
             xerr=[[s["med"] - s["q1"]], [s["q3"] - s["med"]]],
             error_kw=dict(ecolor="black", lw=0.6, capsize=2))
    rng = np.random.default_rng(7 + i)
    axB.scatter(s["spi"], np.full(s["n"], i) + rng.normal(0, 0.10, s["n"]),
                s=14, c="black", alpha=0.55, linewidths=0, zorder=5)
axB.set_xscale("log")
axB.set_yticks(np.arange(len(order_b)))
axB.set_yticklabels([LBL_STACK[m] for m in order_b], fontsize=8.3)
axB.set_xlabel("Fit runtime per iteration (s, log scale)")
axB.set_title("B. Per-implementation runtime distribution", loc="left",
              fontsize=10, fontweight="bold")
axB.grid(axis="x", which="both", linestyle=":", linewidth=0.5, alpha=0.5)

fig.suptitle("Cross-implementation runtime — ds004505, CPU, matched low-iteration budget "
             "(same implementations as the memory comparison)", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_impl_runtime.{ext}", dpi=150, bbox_inches="tight")
print("wrote", HERE / "fig_impl_runtime.pdf")
print("implementations:", {m: round(summ[m]["med"], 3) for m in impls}, "s/iter (median)")
