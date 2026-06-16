"""Render the cross-implementation memory figure from mem_comparison_table.csv.

Panel A (CPU): stacked peak RSS = shared data-load/import floor (baseline, hatched grey) + fit
(delta-RSS, coloured). Panel B (GPU): peak VRAM (each framework's allocator high-water mark).
Reproduces figures/fig_mem_comparison.pdf in the preprint from committed numbers.

Provenance: see PROVENANCE.md (fir jobs 44420385 CPU / 44423654 GPU / 44424365 Fortran-recover).
Run: python plot_mem_comparison.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "mem_comparison_table.csv").open(encoding="utf-8")))

LBL = {
    "amica_python_jax": "AMICA-Python\n(full-batch)",
    "amica_python_jax_chunked": "AMICA-Python\n(chunked)",
    "scott_huberty_torch": "scott-huberty",
    "pyamica_torch": "pyamica",
    "fortran_amica17": "Fortran\nAMICA 1.7",
}
COL = {
    "amica_python_jax": "#1F4E79", "amica_python_jax_chunked": "#5B9BD5",
    "scott_huberty_torch": "#D77A00", "pyamica_torch": "#2A9D8F", "fortran_amica17": "#7030A0",
}
FLOOR = "#DADADA"

cpu = {r["implementation"]: r for r in rows if r["device"] == "cpu"}
gpu = {r["implementation"]: r for r in rows if r["device"] == "gpu"}
cpu_order = [m for m in ["amica_python_jax", "amica_python_jax_chunked", "scott_huberty_torch",
                         "pyamica_torch", "fortran_amica17"] if m in cpu]
gpu_order = [m for m in ["amica_python_jax_chunked", "scott_huberty_torch", "pyamica_torch"] if m in gpu]

fig, (axc, axg) = plt.subplots(1, 2, figsize=(11, 4.9))

# Panel A — stacked: floor (baseline) + fit (delta)
xc = np.arange(len(cpu_order))
peaks = [float(cpu[m]["peak_rss_gb"]) for m in cpu_order]
deltas = [float(cpu[m]["delta_rss_gb"]) for m in cpu_order]
bases = [p - d for p, d in zip(peaks, deltas)]
axc.bar(xc, bases, width=0.66, color=FLOOR, hatch="//", edgecolor="#9a9a9a", linewidth=0.4)
axc.bar(xc, deltas, width=0.66, bottom=bases, color=[COL[m] for m in cpu_order])
for x, p in zip(xc, peaks):
    axc.text(x, p, f" {p:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axc.set_xticks(xc)
axc.set_xticklabels([LBL[m] for m in cpu_order], fontsize=8.3)
axc.set_ylabel("Peak RSS (GB)")
axc.set_title("A. CPU peak memory (RSS)", loc="left", fontweight="bold")
axc.set_ylim(0, max(peaks) * 1.22)
axc.legend(
    handles=[Patch(facecolor=FLOOR, hatch="//", edgecolor="#9a9a9a",
                   label="shared data-load + import floor"),
             Patch(facecolor="#666666", label="fit ($\\Delta$RSS)")],
    fontsize=7.5, loc="upper right", frameon=False,
)
if "amica_python_jax" in cpu_order and "amica_python_jax_chunked" in cpu_order:
    i0, i1 = cpu_order.index("amica_python_jax"), cpu_order.index("amica_python_jax_chunked")
    axc.annotate("", xy=(i1, peaks[i1] + 0.5), xytext=(i0, peaks[i0] + 0.3),
                 arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.6))
    axc.text((i0 + i1) / 2, max(peaks[i0], peaks[i1]) + 1.0, "chunk_size dial",
             ha="center", fontsize=8.3, color="#1F4E79", style="italic")

# Panel B — GPU VRAM
xg = np.arange(len(gpu_order))
vram = [float(gpu[m]["peak_vram_gb"]) for m in gpu_order]
axg.bar(xg, vram, width=0.6, color=[COL[m] for m in gpu_order])
for x, v in zip(xg, vram):
    axg.text(x, v, f" {v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
glabels = [LBL[m].replace("(chunked)", "(auto-chunk)") for m in gpu_order]
axg.set_xticks(xg)
axg.set_xticklabels(glabels, fontsize=8.3)
axg.set_ylabel("Peak VRAM (GB)")
axg.set_title("B. GPU peak memory (VRAM)", loc="left", fontweight="bold")
axg.set_ylim(0, max(vram) * 1.20)

fig.suptitle("Cross-implementation memory — ds004505 sub-01 (C=64, T=785,328, 100 iterations)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_mem_comparison.{ext}", dpi=150)
print("wrote", HERE / "fig_mem_comparison.pdf")
