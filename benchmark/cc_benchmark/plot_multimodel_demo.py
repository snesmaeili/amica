"""Figure for the multi-model AMICA real-EEG demo (ds004505).

Reads mm_demo_sub-NN_M{H}.npz files produced by run_multimodel_demo.py and, per
subject, renders:
  A. model-probability time-course p(h|t) (smoothed) with task-event markers —
     does the active model track the task structure? (Hsu et al. 2018)
  B. final log-likelihood vs number of models H — does H>=2 beat H=1?

Usage (local, after rsync):
  python scripts/cc_benchmark/plot_multimodel_demo.py \
      --root results/multimodel_demo --out-dir results/multimodel_demo/figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SUB_RE = re.compile(r"sub-(\d+)_M(\d+)")
_COLORS = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#d62728", "#17becf"]


def _smooth(x, win):
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _load(root: Path):
    runs = {}
    for f in sorted(root.glob("mm_demo_sub-*_M*.npz")):
        m = _SUB_RE.search(f.name)
        if not m:
            continue
        sid, H = int(m.group(1)), int(m.group(2))
        runs.setdefault(sid, {})[H] = dict(np.load(f, allow_pickle=True))
    return runs


def _plot_subject(sid, by_H, out_dir: Path):
    Hs = sorted(by_H)
    # choose the largest H>=2 (most regimes) for the time-course panel
    H_show = max([h for h in Hs if h >= 2], default=Hs[-1])
    d = by_H[H_show]
    post = np.asarray(d["model_posteriors"], dtype=float)  # (M, T)
    sfreq = float(d["sfreq"])
    M, T = post.shape
    t = np.arange(T) / sfreq
    win = max(1, int(sfreq * 2.0))  # ~2 s smoothing for display

    fig, (ax_p, ax_ll) = plt.subplots(
        1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [3.0, 1.0]}
    )

    # Panel A — p(h|t)
    for h in range(M):
        ax_p.plot(t, _smooth(post[h], win), color=_COLORS[h % len(_COLORS)],
                  lw=1.3, label=f"model {h + 1} (γ={float(d['gm'][h]):.2f})")
    on = np.asarray(d["event_onsets"], dtype=float)
    if on.size:
        ax_p.plot(on, np.full(on.shape, -0.03), "|", color="#444", ms=6, alpha=0.5,
                  label=f"task events (n={on.size})")
    ax_p.set_ylim(-0.06, 1.02)
    ax_p.set_xlabel("time (s)")
    ax_p.set_ylabel("model probability p(h|t)")
    ax_p.set_title(
        f"A. sub-{sid:02d}: {H_show}-model probability time-course (2 s smoothed)",
        loc="left", fontweight="bold",
    )
    ax_p.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    dev = str(d.get("device", "?"))
    ax_p.text(0.01, 0.97, f"device={dev}", transform=ax_p.transAxes, fontsize=7,
              va="top", color="#666")

    # Panel B — LL vs H
    ll = [float(by_H[h]["ll_final"]) for h in Hs]
    ax_ll.plot(Hs, ll, "o-", color="#222")
    for h, y in zip(Hs, ll):
        ax_ll.annotate(f"{y:.3f}", (h, y), textcoords="offset points", xytext=(0, 6),
                       ha="center", fontsize=7)
    ax_ll.set_xticks(Hs)
    ax_ll.set_xlabel("num_models H")
    ax_ll.set_ylabel("final log-likelihood")
    ax_ll.set_title("B. LL vs H", loc="left", fontweight="bold")
    ax_ll.grid(alpha=0.3)

    fig.suptitle(
        f"Multi-model AMICA on ds004505 sub-{sid:02d} — does the active model track the task?",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.13, wspace=0.22)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"fig_multimodel_demo_sub-{sid:02d}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p, dict(H_show=H_show, Hs=Hs, ll=ll, gm=np.round(np.asarray(d["gm"]), 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (args.root / "figures")
    runs = _load(args.root)
    if not runs:
        raise SystemExit(f"No mm_demo_*.npz found under {args.root}")
    print(f"=== multi-model demo: {len(runs)} subject(s) ===")
    for sid in sorted(runs):
        p, summary = _plot_subject(sid, runs[sid], out_dir)
        ll_by_H = {h: round(v, 4) for h, v in zip(summary["Hs"], summary["ll"])}
        better = (len(summary["ll"]) > 1 and summary["ll"][-1] >= summary["ll"][0])
        print(f"  sub-{sid:02d}: LL(H)={ll_by_H}  H>=2 beats H=1: {better}  "
              f"gm(H={summary['H_show']})={summary['gm']}")
        print(f"    wrote {p}")


if __name__ == "__main__":
    main()
