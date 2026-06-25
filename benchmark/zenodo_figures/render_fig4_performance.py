"""Main Figure 4 — computational practicality, memory control, observed scaling.

On an H100, amica-python reaches the evaluated 3,000-iteration solution in the runtime
range of CPU comparators, while explicit chunking bounds memory; CPU execution remains
~2 orders of magnitude slower. Panels (guideline §10): (a) observed quality-cost trade-off,
(b) subject-level total runtime, (c) CPU peak RSS, (d) GPU peak VRAM. CPU and GPU memory
are NOT on a common axis. Reads committed CSVs; full T/C scaling -> Supp S4. No new fits.
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
BENCH = HERE.parent / "cc_benchmark" / "results" / "v3_paper_stage1_cluster" / "benchmark_results.csv"
MEM = Path("D:/amica-validation-workspace/results/mem_compare/mem_comparison_table.csv")

# benchmark method -> (colour, marker, open?, label)
MSTYLE = {
    "AMICA-Python (JAX-GPU)":   (style.AMICA_GPU, "o", False, "amica JAX-GPU"),
    "AMICA-Python (JAX-CPU)":   (style.AMICA_CPU, "s", True,  "amica JAX-CPU"),
    "AMICA-Python (NumPy-CPU)": (style.AMICA_NPY, "^", True,  "amica NumPy-CPU"),
    "Picard":  (style.PICARD,  "s", False, "Picard"),
    "Infomax": (style.INFOMAX, "D", False, "Ext. Infomax"),
    "FastICA": (style.FASTICA, "^", False, "FastICA"),
}
MEM_LABEL = {
    "amica_python_jax": "amica (full)", "amica_python_jax_chunked": "amica (chunked)",
    "scott_huberty_torch": "Scott (torch)", "pyamica_torch": "pyamica", "fortran_amica17": "Fortran 1.7",
}


def main():
    style.set_paper_style()
    df = pd.read_csv(BENCH)
    mem = pd.read_csv(MEM)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    (ax_a, ax_b), (ax_c, ax_d) = axs

    # ---- a: quality-cost (median fit time vs mean MIR) ----
    for meth, (col, mk, op, lab) in MSTYLE.items():
        g = df[df.method == meth]
        if g.empty:
            continue
        t = g.fit_runtime_s.dropna(); mir = g.mir_kbits_s.dropna()
        tx = np.median(t); my = np.mean(mir)
        ax_a.errorbar(tx, my, xerr=[[tx - np.percentile(t, 25)], [np.percentile(t, 75) - tx]],
                      yerr=mir.std() / np.sqrt(len(mir)), fmt=mk, ms=7, color=col, label=lab,
                      mfc=("none" if op else col), mec=col, ecolor=col, elinewidth=1, capsize=2, zorder=3)
    ax_a.legend(fontsize=6.2, loc="lower center", frameon=False, ncol=2,
                handletextpad=0.3, columnspacing=0.9, borderpad=0.2)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("median fit time (s)"); ax_a.set_ylabel("mean MIR (kbit/s)")
    ax_a.set_title("a  Quality-cost trade-off", loc="left", fontweight="bold", fontsize=9.5)
    ax_a.text(0.5, -0.30, "amica AMICA fits run to the 3,000-iteration cap; comparators use their own stopping criteria.",
              transform=ax_a.transAxes, ha="center", fontsize=6.2, color="#666", style="italic")

    # ---- b: subject-level total runtime (log x) ----
    order = list(MSTYLE)
    for i, meth in enumerate(order):
        g = df[df.method == meth]
        if g.empty:
            continue
        col, mk, op, lab = MSTYLE[meth]
        y = len(order) - 1 - i
        t = g.fit_runtime_s.dropna().to_numpy()
        jit = (np.random.default_rng(0).random(len(t)) - 0.5) * 0.5
        ax_b.scatter(t, np.full(len(t), y) + jit, s=8, color=col, alpha=0.4, edgecolors="none")
        ax_b.plot([np.percentile(t, 25), np.percentile(t, 75)], [y, y], color="k", lw=1.2)
        ax_b.plot(np.median(t), y, mk, ms=6, color=col, mfc=("none" if op else col), mec="k", mew=0.5)
    ax_b.set_xscale("log"); ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels([MSTYLE[m][3] for m in reversed(order)], fontsize=7.5)
    ax_b.set_xlabel("fit time per subject (s)")
    ax_b.set_title("b  Subject-level runtime", loc="left", fontweight="bold", fontsize=9.5)

    # ---- c: CPU peak RSS (incremental fit RSS) ----
    cpu = mem[mem.device == "cpu"].sort_values("delta_rss_gb")
    yc = range(len(cpu))
    cols_c = [style.AMICA_BLUE if "amica" in i else style.GREY for i in cpu.implementation]
    ax_c.barh(list(yc), cpu.delta_rss_gb, color=cols_c, alpha=0.85, height=0.6)
    for y, (_, r) in zip(yc, cpu.iterrows()):
        tag = " (>= upper bound)" if r.implementation == "pyamica_torch" else ""
        ax_c.text(r.delta_rss_gb + 0.1, y, f"{r.delta_rss_gb:.1f}{tag}", va="center", fontsize=6.8)
    ax_c.set_yticks(list(yc)); ax_c.set_yticklabels([MEM_LABEL.get(i, i) for i in cpu.implementation], fontsize=7.5)
    ax_c.set_xlabel("incremental fit RSS (GB), CPU"); ax_c.set_xlim(0, cpu.delta_rss_gb.max() * 1.35)
    ax_c.set_title("c  CPU memory (1 subject)", loc="left", fontweight="bold", fontsize=9.5)

    # ---- d: GPU peak VRAM ----
    gpu = mem[mem.device == "gpu"].dropna(subset=["peak_vram_gb"]).sort_values("peak_vram_gb")
    yd = range(len(gpu))
    cols_d = [style.AMICA_BLUE if "amica" in i else style.GREY for i in gpu.implementation]
    ax_d.barh(list(yd), gpu.peak_vram_gb, color=cols_d, alpha=0.85, height=0.55)
    for y, (_, r) in zip(yd, gpu.iterrows()):
        ax_d.text(r.peak_vram_gb + 0.3, y, f"{r.peak_vram_gb:.1f}", va="center", fontsize=6.8)
    ax_d.set_yticks(list(yd)); ax_d.set_yticklabels([MEM_LABEL.get(i, i) for i in gpu.implementation], fontsize=7.5)
    ax_d.set_xlabel("peak VRAM (GB), H100"); ax_d.set_xlim(0, gpu.peak_vram_gb.max() * 1.25)
    ax_d.set_title("d  GPU memory (1 subject)", loc="left", fontweight="bold", fontsize=9.5)

    fig.suptitle("amica-python: practical H100 runtime + chunked-bounded memory "
                 "(ds004505; all AMICA at the 3,000-iter budget)", fontsize=9.4, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = style.save_vector(fig, HERE / "out" / "fig4_performance.pdf")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
