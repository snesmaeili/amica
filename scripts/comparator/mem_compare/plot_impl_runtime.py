"""Cross-implementation runtime/speed figure (companion to fig_mem_comparison.pdf).

Same AMICA implementations as the memory comparison -- amica-python (JAX, full-batch + chunked),
scott-huberty (PyTorch), pyamica (PyTorch), Fortran AMICA 1.7 -- on ds004505. Runtime/speed instead
of memory. Three panels:

  Panel A (CPU): final log-likelihood (the AMICA objective; ~identical across implementations -> same
                 optimum) vs per-iteration fit runtime, on the reference subject sub-01. ll is a
                 per-sample likelihood, comparable only WITHIN a subject, hence sub-01 only.
  Panel B (CPU): per-implementation per-iteration runtime distribution across subjects (median + IQR,
                 per-subject dots). Runtime is per ITERATION (fit_time_s / n_iter) so the 100-iter
                 (sub-01) and 60-iter (sub-02..06) runs are comparable.
  Panel C (GPU): steady-state per-iteration runtime on one H100, for the 3 implementations with a GPU
                 path. The fixed one-time cost is removed: the PyTorch implementations via the 2-point
                 method (T_600 - T_100)/500; amica-python's 600-iter run reused XLA's compilation
                 cache (no recompile) so its T_600/600 is already JIT-free steady state. A RAW 100-iter
                 GPU fit is ~90% JIT compile for JAX and would badly understate it.

With --json-root PATH (the workspace results dir holding mem_compare/, mem_multisubj/, rt_gpu_100/,
rt_gpu_600/) this rebuilds impl_runtime_table.csv + impl_runtime_gpu.csv; otherwise it plots the
committed CSVs.
Run: python plot_impl_runtime.py --json-root D:/amica-validation-workspace/results
     python plot_impl_runtime.py            # from committed CSVs
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
MARK = {"amica_python_jax": "o", "amica_python_jax_chunked": "o", "scott_huberty_torch": "s",
        "pyamica_torch": "D", "fortran_amica17": "^"}
ORDER = ["amica_python_jax", "amica_python_jax_chunked", "scott_huberty_torch",
         "pyamica_torch", "fortran_amica17"]
GPU_ORDER = ["amica_python_jax_chunked", "scott_huberty_torch", "pyamica_torch"]


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
    # GPU 2-point
    t = {100: {}, 600: {}}
    for it in (100, 600):
        for jf in glob.glob(str(root / f"rt_gpu_{it}" / "*_result.json")):
            a = json.load(open(jf))
            if a.get("implementation") in GPU_ORDER and a.get("fit_time_s"):
                t[it][a["implementation"]] = a["fit_time_s"]
    grows = []
    for im in GPU_ORDER:
        if im in t[100] and im in t[600]:
            two_pt = (t[600][im] - t[100][im]) / 500.0
            spi = two_pt if two_pt > 0 else t[600][im] / 600.0  # else: JAX XLA-cached 600-iter
            grows.append(dict(implementation=im, t100_s=t[100][im], t600_s=t[600][im],
                              gpu_s_per_iter=spi, method="2point" if two_pt > 0 else "cached600"))
    with (HERE / "impl_runtime_gpu.csv").open("w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["implementation", "t100_s", "t600_s",
                                          "gpu_s_per_iter", "method"])
        w.writeheader(); w.writerows(grows)
    return rows, grows


def _load():
    rows = list(csv.DictReader((HERE / "impl_runtime_table.csv").open(encoding="utf-8")))
    for r in rows:
        for k in ("ll_final", "fit_time_s", "s_per_iter"):
            r[k] = float(r[k])
        r["n_iter"] = int(r["n_iter"])
    grows = list(csv.DictReader((HERE / "impl_runtime_gpu.csv").open(encoding="utf-8")))
    for r in grows:
        for k in ("t100_s", "t600_s", "gpu_s_per_iter"):
            r[k] = float(r[k])
    return rows, grows


rows, grows = _build(ROOT) if ROOT else _load()
gpu = {r["implementation"]: r for r in grows}

impls = [m for m in ORDER if any(r["implementation"] == m for r in rows)]
summ = {}
for m in impls:
    spi = np.array([r["s_per_iter"] for r in rows if r["implementation"] == m])
    summ[m] = dict(med=float(np.median(spi)), q1=float(np.percentile(spi, 25)),
                   q3=float(np.percentile(spi, 75)), n=len(spi), spi=spi)

fig = plt.figure(figsize=(16.5, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 0.95], wspace=0.34)

# --- Panel A: CPU quality-cost (sub-01) ---
axA = fig.add_subplot(gs[0, 0])
sub01 = {r["implementation"]: r for r in rows if r["subject"] == "sub-01"}
for m in impls:
    if m not in sub01:
        continue
    r = sub01[m]
    axA.scatter(r["s_per_iter"], r["ll_final"], marker=MARK[m], color=COL[m], s=130,
                edgecolor="black", linewidths=0.6, alpha=0.95, zorder=3, label=LBL_INLINE[m])
ll01 = [sub01[m]["ll_final"] for m in impls if m in sub01]
rt01 = [sub01[m]["s_per_iter"] for m in impls if m in sub01]
axA.set_xscale("log")
axA.set_xlabel("Fit runtime per iteration (s, log scale)")
axA.set_ylabel("Final log-likelihood (AMICA objective)")
axA.set_ylim(float(np.mean(ll01)) - 0.04, float(np.mean(ll01)) + 0.04)
axA.annotate("all five reach the same\noptimum ($\\Delta$ll $<10^{-3}$)", xy=(0.5, 0.85),
             xycoords="axes fraction", ha="center", fontsize=8.3, color="0.35", fontstyle="italic")
axA.annotate("faster", xy=(min(rt01) * 0.78, np.mean(ll01) - 0.030),
             xytext=(min(rt01) * 2.2, np.mean(ll01) - 0.030), ha="center", va="center",
             fontsize=9, color="0.35", fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0))
axA.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
axA.set_title("A. CPU quality–cost (sub-01)", loc="left", fontsize=9.5, fontweight="bold")
axA.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.8",
           fontsize=7.8, handlelength=1.2, borderpad=0.35, labelspacing=0.25)

# --- Panel B: CPU runtime distribution ---
axB = fig.add_subplot(gs[0, 1])
order_b = sorted(impls, key=lambda m: summ[m]["med"])
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
axB.set_yticklabels([LBL_STACK[m] for m in order_b], fontsize=8.0)
axB.set_xlabel("Fit runtime per iteration (s, log scale)")
axB.set_title("B. CPU runtime distribution", loc="left", fontsize=9.5, fontweight="bold")
axB.grid(axis="x", which="both", linestyle=":", linewidth=0.5, alpha=0.5)

# --- Panel C: GPU steady-state per iteration (H100) ---
axC = fig.add_subplot(gs[0, 2])
gorder = sorted([m for m in GPU_ORDER if m in gpu], key=lambda m: gpu[m]["gpu_s_per_iter"])
for i, m in enumerate(gorder):
    v = gpu[m]["gpu_s_per_iter"]
    axC.barh(i, v, height=0.6, color=COL[m], edgecolor="black", linewidth=0.4, alpha=0.8)
    axC.text(v, i, f" {v * 1000:.0f} ms", va="center", ha="left", fontsize=8.5, fontweight="bold")
axC.set_yticks(np.arange(len(gorder)))
axC.set_yticklabels([LBL_STACK[m].replace("(chunked)", "(JAX-GPU)") for m in gorder], fontsize=8.0)
axC.set_xlabel("Steady-state runtime per iteration (s)")
axC.set_xlim(0, max(gpu[m]["gpu_s_per_iter"] for m in gorder) * 1.32)
axC.set_title("C. GPU steady-state per iteration (H100)", loc="left", fontsize=9.5, fontweight="bold")
axC.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

fig.suptitle("Cross-implementation runtime — ds004505 (same implementations as the memory comparison)",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_impl_runtime.{ext}", dpi=150, bbox_inches="tight")
print("wrote", HERE / "fig_impl_runtime.pdf")
print("CPU s/iter (median):", {m: round(summ[m]["med"], 3) for m in impls})
print("GPU s/iter (steady):", {m: round(gpu[m]["gpu_s_per_iter"], 4) for m in gorder})
