"""Build the Fortran AMICA 1.7 <-> amica-python parity figure + summary table.

Reads one or more ``parity.json`` files (from run_fortran_parity.py compare) and
produces:
  - a 2-panel figure (LL-trajectory overlay; log-scale deviation bars), PDF + PNG
  - a LaTeX table snippet + a markdown table on stdout

Run LOCALLY on JSONs pulled from the cluster (never on a login node).

    python make_parity_figure.py \
        synth6=results/parity/mklfix50k.parity.json \
        ds004505_sub01=results/parity/ds004505_sub01.parity.json \
        --out results/parity/fig_fortran_parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252, which can't encode Δ/← in the summary table.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load(label_path):
    label, path = label_path.split("=", 1)
    d = json.loads(Path(path).read_text())
    return label, d


def _traj(d, side):
    """Full trajectory if present, else the first-5 fallback."""
    full = d.get(f"ll_full_{side}")
    if full:
        return np.asarray(full, float)
    return np.asarray(d.get(f"ll_first5_{side}", []), float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", help="label=path/to/parity.json")
    ap.add_argument("--out", default="fig_fortran_parity",
                    help="output stem (writes .pdf, .png, .tex)")
    args = ap.parse_args()

    runs = [_load(lp) for lp in args.datasets]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(runs)))

    # --- Panel A: LL trajectory overlay (Fortran solid, amica-python dashed) ---
    for (label, d), c in zip(runs, colors):
        lf, lp = _traj(d, "fortran"), _traj(d, "python")
        xf, xp = np.arange(1, lf.size + 1), np.arange(1, lp.size + 1)
        axA.plot(xf, lf, "-", color=c, lw=1.8, label=f"{label} — Fortran 1.7")
        axA.plot(xp, lp, "--", color=c, lw=1.2, label=f"{label} — amica-python")
    axA.set_xlabel("iteration")
    axA.set_ylabel("log-likelihood (nats / sample / channel)")
    axA.set_title("LL trajectory (curves overlie)")
    axA.legend(fontsize=7, loc="lower right")
    axA.grid(alpha=0.3)

    # --- Panel B: log-scale deviation metrics (smaller = better parity) ---
    metrics = [
        ("iter-0 |ΔLL|", lambda d: d.get("ll_iter0_abs_diff", np.nan)),
        ("final |ΔLL|", lambda d: d.get("abs_ll_delta", np.nan)),
        ("1 − source |r|", lambda d: 1.0 - d.get("matched_source_abs_r_mean", np.nan)),
        ("1 − W |r|", lambda d: 1.0 - d.get("W_matched_abs_r_mean", np.nan)),
    ]
    n_m, n_r = len(metrics), len(runs)
    y = np.arange(n_m)
    h = 0.8 / max(n_r, 1)
    for j, ((label, d), c) in enumerate(zip(runs, colors)):
        vals = [max(float(f(d)), 1e-16) for _, f in metrics]
        axB.barh(y + j * h, vals, height=h, color=c, label=label)
    axB.set_yticks(y + 0.4 - h / 2)
    axB.set_yticklabels([m[0] for m in metrics])
    axB.set_xscale("log")
    axB.axvline(1e-6, color="0.4", ls=":", lw=1)
    axB.set_xlabel("absolute deviation (log scale)")
    axB.set_title("Parity deviations (← exact)")
    axB.legend(fontsize=7, loc="lower right")
    axB.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[fig] wrote {out.with_suffix('.pdf')} and .png")

    # --- summary tables ---
    cols = [
        ("dataset", lambda lbl, d: lbl),
        ("n_ch", lambda lbl, d: d["config"].get("n_channels", "?")),
        ("n_samp", lambda lbl, d: d["config"].get("n_samples", "?")),
        ("iters", lambda lbl, d: d.get("fortran_n_iter", "?")),
        ("iter0 |ΔLL|", lambda lbl, d: f"{d.get('ll_iter0_abs_diff', float('nan')):.2e}"),
        ("final |ΔLL|", lambda lbl, d: f"{d.get('abs_ll_delta', float('nan')):.2e}"),
        ("rel ΔLL %", lambda lbl, d: f"{d.get('rel_ll_delta_pct', float('nan')):.2e}"),
        ("source |r|", lambda lbl, d: f"{d.get('matched_source_abs_r_mean', float('nan')):.10f}"),
        ("W |r|", lambda lbl, d: f"{d.get('W_matched_abs_r_mean', float('nan')):.10f}"),
    ]
    header = [c[0] for c in cols]
    rows = [[str(c[1](lbl, d)) for c in cols] for lbl, d in runs]

    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")

    tex = out.with_suffix(".tex")
    with tex.open("w", newline="\n", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\hline\n")
        fh.write(" & ".join(h.replace("%", "\\%").replace("|", "$|$").replace("Δ", "$\\Delta$")
                            for h in header) + " \\\\\n\\hline\n")
        for r in rows:
            fh.write(" & ".join(str(x) for x in r) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    print(f"[tex] wrote {tex}")


if __name__ == "__main__":
    main()
