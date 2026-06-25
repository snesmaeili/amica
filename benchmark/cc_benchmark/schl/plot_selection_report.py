"""Render the SCHL criterion curves from a run_auto_select.py SelectionReport JSON.

Three panels per subject: (a) rank N -- full-dimensional held-out LL vs N; (b) model order
H -- held-out DeltaLL increment vs the phase-surrogate null P95 (the stop rule); (c)
rejection -- held-out LL vs rejsig + the surrogate-null floor. Used to eyeball the pilot
results (does the criterion pick M>1, are the curves + nulls sane) before the full grid.

Usage: python plot_selection_report.py report.json [out.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ikeys(d):
    """{'32': v, '64': v} -> sorted [(32, v), ...] (int keys)."""
    return sorted(((int(k), v) for k, v in d.items()), key=lambda t: t[0])


def _rejkey(k):
    """JSON rejsig key -> float or None ('null'/'None'/'off' -> off baseline)."""
    return None if str(k).lower() in ("null", "none", "off") else float(k)


def render(report: dict, out_png: Path):
    sel = report.get("selected", {})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # (a) rank
    rk = report.get("rank_info", {}).get("full_heldout_ll", {})
    if rk:
        xs, ys = zip(*_ikeys(rk))
        axs[0].plot(xs, ys, "o-", color="#08306b")
        axs[0].axvline(sel.get("n_components"), color="#d62728", ls="--", lw=1)
    axs[0].set_xlabel("PCA rank N"); axs[0].set_ylabel("full-data held-out LL")
    axs[0].set_title(f"a  Rank -> N*={sel.get('n_components')}", loc="left", fontweight="bold", fontsize=9.5)

    # (b) model order: held-out DeltaLL increment vs surrogate null P95
    rows = report.get("model_order_info", {}).get("increments", [])
    if rows:
        H = [r["H"] for r in rows]
        inc = [r.get("increment") for r in rows]
        null = [r.get("null_p95") for r in rows]
        axs[1].plot(H, inc, "o-", color="#08306b", label="held-out $\\Delta$LL incr.")
        axs[1].plot(H, null, "s--", color="#9aa3ad", label="surrogate null P95")
        axs[1].axhline(0, color="k", lw=0.6)
        axs[1].axvline(sel.get("num_models"), color="#d62728", ls="--", lw=1)
        axs[1].legend(fontsize=7.5, frameon=False)
    axs[1].set_xlabel("model order H"); axs[1].set_ylabel("held-out $\\Delta$LL increment")
    axs[1].set_title(f"b  Model order -> M*={sel.get('num_models')}", loc="left", fontweight="bold", fontsize=9.5)

    # (c) rejection profile
    rej = report.get("rejection_info", {})
    hl = rej.get("heldout_ll", {})
    if hl:
        pts = sorted(((_rejkey(k), v) for k, v in hl.items()),
                     key=lambda t: (t[0] is None, -(t[0] or 0)))
        labs = ["off" if k is None else f"{k:g}" for k, _ in pts]
        vals = [v for _, v in pts]
        axs[2].plot(range(len(vals)), vals, "o-", color="#08306b")
        axs[2].set_xticks(range(len(labs))); axs[2].set_xticklabels(labs)
        if rej.get("surrogate_null95") is not None and rej.get("off_ll") is not None:
            axs[2].axhline(rej["off_ll"] + max(rej["surrogate_null95"], 0), color="#9aa3ad",
                           ls="--", lw=1, label="off + surrogate null")
            axs[2].legend(fontsize=7.5, frameon=False)
    rs = sel.get("rejsig")
    axs[2].set_xlabel("rejsig"); axs[2].set_ylabel("median held-out LL")
    axs[2].set_title(f"c  Rejection -> rejsig*={'off' if rs is None else rs}",
                     loc="left", fontweight="bold", fontsize=9.5)

    ke = sel.get("kappa_effective")
    fig.suptitle(f"SCHL selection: {report.get('dataset','?')} sub-{report.get('subject','?')} "
                 f"(N*={sel.get('n_components')}, M*={sel.get('num_models')}, "
                 f"rejsig*={'off' if rs is None else rs}, kappa_eff={ke:.0f})" if ke else "SCHL selection",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    rep = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).with_suffix(".png")
    render(rep, out)
