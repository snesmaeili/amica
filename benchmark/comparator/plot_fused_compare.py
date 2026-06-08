"""Stage 3E — 3-way (vs 4-bar) fused-AMICA comparison figure + summary.

Builds a device-honest runtime + peak-memory comparison from comparator result
JSONs, grouping by (implementation, device) so AMICA-Python's CPU and GPU runs
are SEPARATE series (the existing paper-figure `plot_comparator_runtime_memory`
groups by method only and would collapse them).

Series rendered (whichever are present):
  - AMICA-classic (CPU)  : the OLD pilot recompute path  (--old-root)
  - AMICA-fused   (CPU)  : Stage 3D fused E-step, CPU     (cmp_fused/amica_cpu)
  - AMICA-fused   (GPU)  : Stage 3D fused E-step, H100    (cmp_fused/amica_gpu)
  - pyamica       (CPU)  : DerAndereJohannes/pyamica      (cmp_fused/pyamica_cpu)

Usage (local, after rsync of the fir results):
  python scripts/comparator/plot_fused_compare.py \
      --new-root results/comparator/cmp_fused \
      --old-root results/comparator_pilot_cluster \
      --out-dir  results/comparator/cmp_fused/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (implementation, device) -> (display label, color, sort order)
SERIES = {
    ("amica_python_jax", "cpu", "classic"): ("AMICA-classic\n(CPU)", "#9ecae1", 0),
    ("amica_python_jax", "cpu", "fused"): ("AMICA-fused\n(CPU)", "#3182bd", 1),
    ("pyamica_torch", "cpu", "fused"): ("pyamica\n(CPU)", "#e6550d", 2),
    ("amica_python_jax", "gpu", "fused"): ("AMICA-fused\n(GPU)", "#31a354", 3),
}


def _load_one(path: Path, variant: str) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if "error" in d or "fit_time_s" not in d:
        return None
    impl = d.get("implementation", "")
    device = str(d.get("device", "cpu")).lower()
    key = (impl, device, variant)
    if key not in SERIES:
        return None
    # subject id from the path (…/ds004505_sub-NN/…)
    sub = next((p for p in path.parts if "sub-" in p), path.parent.name)
    return {
        "key": key,
        "label": SERIES[key][0],
        "color": SERIES[key][1],
        "order": SERIES[key][2],
        "subject": sub,
        "fit_time_s": float(d["fit_time_s"]),
        "peak_rss_gb": float(d.get("peak_rss_gb", float("nan"))),
        "n_iter": int(d.get("n_iter", 0)),
    }


def collect(new_root: Path, old_root: Path | None) -> list[dict]:
    rows: list[dict] = []
    # New fused series
    for series_dir, _impl in (("amica_cpu", "amica_python_jax"),
                              ("amica_gpu", "amica_python_jax"),
                              ("pyamica_cpu", "pyamica_torch")):
        for p in (new_root / series_dir).glob("**/*_result.json"):
            r = _load_one(p, "fused")
            if r:
                rows.append(r)
    # Old pilot baseline (classic CPU) — only the amica_python_jax rows
    if old_root and old_root.exists():
        for p in old_root.glob("**/amica_python_jax_*_result.json"):
            r = _load_one(p, "classic")
            if r:
                rows.append(r)
    return rows


def _med_iqr(vals: list[float]) -> tuple[float, float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def render(rows: list[dict], out_dir: Path) -> Path | None:
    if not rows:
        print("[plot_fused_compare] no rows collected — nothing to render")
        return None
    # Group by series key, preserve defined order
    keys = sorted({r["key"] for r in rows}, key=lambda k: SERIES[k][2])
    labels = [SERIES[k][0] for k in keys]
    colors = [SERIES[k][1] for k in keys]
    rng = np.random.default_rng(0)

    fig, (ax_rt, ax_mem) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(keys))

    for col, ax, metric, ylab, title, fmt, logy in (
        (0, ax_rt, "fit_time_s", "Fit runtime (s, log)", "A. Fit runtime", "{:.0f}s", True),
        (1, ax_mem, "peak_rss_gb", "Peak memory (GB)", "B. Peak memory", "{:.2f} GB", False),
    ):
        for i, k in enumerate(keys):
            vals = [r[metric] for r in rows if r["key"] == k]
            med, q1, q3 = _med_iqr(vals)
            ax.bar(i, med, width=0.62, color=colors[i], alpha=0.88, zorder=2)
            arr = np.asarray(vals, dtype=float)
            ax.scatter(i + rng.uniform(-0.12, 0.12, arr.size), arr, s=20,
                       color="#222", zorder=3)
            if np.isfinite(med):
                ax.text(i, q3, " " + fmt.format(med), ha="center", va="bottom", fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)

    n_iter = int(np.median([r["n_iter"] for r in rows if r["n_iter"]]) or 0)
    fig.suptitle(
        f"AMICA-Python fused E-step vs pyamica — ds004505, 10-min crop, {n_iter} iter "
        "(bars = median across subjects; dots = per-subject)",
        fontsize=10, fontweight="bold", y=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.16, wspace=0.26)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_fused_compare_runtime_memory.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize(rows: list[dict]) -> None:
    print("\n=== Stage 3E fused-comparison summary ===")
    meds: dict[tuple, tuple[float, float, int]] = {}
    for k in sorted({r["key"] for r in rows}, key=lambda kk: SERIES[kk][2]):
        rt = [r["fit_time_s"] for r in rows if r["key"] == k]
        mem = [r["peak_rss_gb"] for r in rows if r["key"] == k]
        n = len([r for r in rows if r["key"] == k])
        med_rt, _, _ = _med_iqr(rt)
        med_mem, _, _ = _med_iqr(mem)
        meds[k] = (med_rt, med_mem, n)
        lbl = SERIES[k][0].replace("\n", " ")
        print(f"  {lbl:24s}  median runtime={med_rt:7.1f}s   median mem={med_mem:5.2f}GB   (n={n})")

    ac = ("amica_python_jax", "cpu", "classic")
    af = ("amica_python_jax", "cpu", "fused")
    pf = ("pyamica_torch", "cpu", "fused")
    ag = ("amica_python_jax", "gpu", "fused")
    if ac in meds and af in meds and meds[ac][0] and meds[af][0]:
        print(f"\n  fused-CPU vs classic-CPU : {meds[ac][0] / meds[af][0]:.2f}x "
              f"({meds[ac][0]:.0f}s -> {meds[af][0]:.0f}s)")
    if af in meds and pf in meds and meds[pf][0]:
        d = meds[af][0] - meds[pf][0]
        verb = "faster" if d < 0 else "slower"
        print(f"  fused-CPU vs pyamica-CPU : AMICA {abs(d):.0f}s {verb} "
              f"({meds[af][0]:.0f}s vs {meds[pf][0]:.0f}s)")
    if ag in meds and pf in meds and meds[ag][0]:
        print(f"  fused-GPU vs pyamica-CPU : {meds[pf][0] / meds[ag][0]:.1f}x "
              f"({meds[pf][0]:.0f}s -> {meds[ag][0]:.0f}s)")
    if af in meds and pf in meds and meds[af][1] and meds[pf][1]:
        print(f"  memory: AMICA {meds[af][1]:.2f}GB vs pyamica {meds[pf][1]:.2f}GB "
              f"({meds[pf][1] / meds[af][1]:.1f}x less)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True,
                    help="local dir with cmp_fused/{amica_cpu,amica_gpu,pyamica_cpu}/")
    ap.add_argument("--old-root", type=Path, default=None,
                    help="optional pilot baseline dir (classic CPU amica_python_jax)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (args.new_root / "figures")
    rows = collect(args.new_root, args.old_root)
    summarize(rows)
    p = render(rows, out_dir)
    if p:
        print(f"\n[plot_fused_compare] wrote {p}")


if __name__ == "__main__":
    main()
