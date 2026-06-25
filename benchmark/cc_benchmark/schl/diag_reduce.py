"""SCHL diagnostic REDUCE (run LOCALLY after rsync): combine diag_fit cells -> M1-M5.

For one (subject, budget): reads the `real`, `surr_*`, and `ho_*` cell JSONs and computes
the held-out noise floor (M1/M2) and the in-sample surrogate-calibrated E(H) SNR + knee
(M3/M4). Compare across the two budgets for the convergence check (M5).

Usage: python diag_reduce.py --dir <cells_dir> --dataset ds004505 --subject 1 --budget 2000 \
                             --out diag_sub01_B2000.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _knee_fraction(E, f):
    Hs = sorted(E)
    if len(Hs) < 2:
        return None
    inc = {Hs[i]: E[Hs[i]] - E[Hs[i - 1]] for i in range(1, len(Hs))}
    base = inc[Hs[1]]
    if base <= 0:
        return 1
    M = 1
    for h in Hs[1:]:
        if inc[h] >= f * base:
            M = h
        else:
            break
    return M


def _knee_maxchord(E):
    Hs = sorted(E)
    if len(Hs) < 3:
        return Hs[-1] if Hs else None
    x = np.array(Hs, float); y = np.array([E[h] for h in Hs], float)
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / (np.hypot(y1 - y0, x1 - x0) + 1e-12)
    return int(x[int(np.argmax(d))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="dir of diag_fit cell JSONs")
    ap.add_argument("--dataset", default="ds004505")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--knee-fracs", type=float, nargs="+", default=[0.25, 0.33, 0.5])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from amica_python.benchmark.stationarity import delta_ll

    pat = f"{a.dir}/cell_{a.dataset}_sub{a.subject:02d}_B{a.budget}_*.json"
    cells = [json.load(open(f)) for f in glob.glob(pat)]
    if not cells:
        raise SystemExit(f"no cells matched {pat}")
    real = next(c for c in cells if c["kind"] == "insample")
    surr = [c for c in cells if c["kind"] == "insample_surr"]
    ho = [c for c in cells if c["kind"] == "heldout"]

    ll_real = {int(k): v for k, v in real["ll"].items()}
    Hs = sorted(ll_real)
    d_real = {H: delta_ll(ll_real)[H]["delta"] for H in Hs}
    d_surr = [{H: delta_ll({int(k): v for k, v in s["ll"].items()})[H]["delta"] for H in Hs} for s in surr]
    E = {H: d_real[H] - float(np.nanmean([s[H] for s in d_surr])) for H in Hs}
    sd_surr = {H: (float(np.nanstd([s[H] for s in d_surr], ddof=1)) if len(d_surr) > 1 else float("nan")) for H in Hs}
    snr_E = {H: (E[H] / sd_surr[H] if sd_surr[H] and np.isfinite(sd_surr[H]) and sd_surr[H] > 0 else None) for H in Hs}
    surr2 = float(np.nanmean([s[2] for s in d_surr])) if 2 in Hs and d_surr else float("nan")
    ratio_H2 = float(d_real[2] / surr2) if (2 in Hs and np.isfinite(surr2) and surr2 != 0) else None
    p95_surr2 = float(np.nanpercentile([s[2] for s in d_surr], 95)) if (2 in Hs and len(d_surr) > 1) else float("nan")
    warranted = bool(2 in Hs and E[2] > max(p95_surr2, 0.0))

    ho_incr = [x for c in ho for x in c["increments"]]
    ho_sd = float(np.nanstd(ho_incr, ddof=1)) if len(ho_incr) > 1 else float("nan")
    ho_mean = float(np.nanmean(ho_incr)) if ho_incr else float("nan")

    out = dict(
        dataset=a.dataset, subject=int(a.subject), budget=int(a.budget),
        n_cells=dict(real=1, surr=len(surr), ho=len(ho)),
        insample=dict(ll_real=ll_real, delta_real=d_real, E=E, sd_surr=sd_surr, SNR_E=snr_E,
                      ratio_H2=ratio_H2, p95_surr2=p95_surr2, warranted=warranted,
                      knee_fraction={f"{f:g}": _knee_fraction(E, f) for f in a.knee_fracs},
                      knee_maxchord=_knee_maxchord(E)),
        heldout=dict(heldout_h=(ho[0]["heldout_h"] if ho else None), n=len(ho_incr),
                     mean=ho_mean, sd=ho_sd,
                     signal_over_sd=(ho_mean / ho_sd if ho_sd and np.isfinite(ho_sd) else None)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float), newline="\n")
    print(f"sub-{a.subject:02d} B={a.budget}: warranted={warranted} SNR_E(2)={snr_E.get(2)} "
          f"ratio_H2={ratio_H2} knee={out['insample']['knee_fraction']} | "
          f"held-out signal/sd={out['heldout']['signal_over_sd']}")


if __name__ == "__main__":
    main()
