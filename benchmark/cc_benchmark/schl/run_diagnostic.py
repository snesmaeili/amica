"""SCHL Stage-A diagnostic: held-out noise floor vs in-sample surrogate-calibrated E(H).

Decides whether the model-order axis should switch from the (broken) held-out DeltaLL
criterion to the in-sample surrogate-calibrated excess E(H) = DeltaLL_real - DeltaLL_surr.
For ONE subject at ONE iteration budget B, computes from the SAME fits:

  in-sample arm (the candidate criterion):
    delta_real[H]   = LL_real(H) - LL_real(1)            (best-of-S seeds; stationarity.delta_ll)
    delta_surr_j[H] = LL_surr_j(H) - LL_surr_j(1)        (n_surr shared phase surrogates)
    E[H]            = delta_real[H] - mean_j delta_surr_j[H]
    SNR_E[H]        = E[H] / std_j delta_surr_j[H]        ; ratio_H2 = real/surr at H=2
    knee            = inc_E fraction rule (f) + max-chord-distance cross-check
  held-out arm (the noise floor that broke the pilot):
    per-step increment Lho(H)-Lho(H-1) over (fold x seed) at H in heldout_h -> sd (M1), mean (M2)
  convergence (M5): caller runs B=600 and B=2000 as separate jobs; compare LL_real[H] across.

Decision (combine the two budgets offline): if held-out signal/sd > 1.64 at B=2000 -> robustified
held-out alive; elif SNR_E(2) >~ 3 and ratio_H2 >= 5 and knee at H<H_max -> adopt in-sample E(H);
elif SNR_E(2) < 2 -> per-subject too weak, retreat to cohort-level warranted-test only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _insample_best_of_S(fit, X, H, seeds, **kw):
    """Best-of-S in-sample final LL at model order H (highest-likelihood basin)."""
    lls = []
    for s in seeds:
        try:
            lls.append(float(fit(X, num_models=H, seed=int(s), **kw).log_likelihood[-1]))
        except Exception:
            lls.append(np.nan)
    return float(np.nanmax(lls)) if np.any(np.isfinite(lls)) else np.nan


def _knee_fraction(E, f):
    """Largest H with inc_E(H) >= f*inc_E(2), unbroken chain from H=2 (None if not warranted)."""
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
    """H of maximum perpendicular distance of E(H) from the chord (endpoints) -- elbow."""
    Hs = sorted(E)
    if len(Hs) < 3:
        return Hs[-1] if Hs else None
    x = np.array(Hs, float); y = np.array([E[h] for h in Hs], float)
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / (np.hypot(y1 - y0, x1 - x0) + 1e-12)
    return int(x[int(np.argmax(d))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds004505")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--input-level", default="bids")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--h-max", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-iter", type=int, required=True, help="the iteration budget B for this run")
    ap.add_argument("--n-surr", type=int, default=5)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--heldout-h", type=int, nargs="+", default=[2, 3], help="H to measure held-out noise at")
    ap.add_argument("--heldout-folds", type=int, default=5)
    ap.add_argument("--knee-fracs", type=float, nargs="+", default=[0.25, 0.33, 0.5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    from amica_python.benchmark.runner import load_data, preprocess
    from amica_python.benchmark.stationarity import delta_ll
    from amica_python.selector import _fit_amica, heldout_loglik, phase_randomize, block_folds

    raw = load_data(args.dataset, args.subject, input_level=args.input_level)
    raw.load_data()
    if args.sfreq and abs(float(raw.info["sfreq"]) - args.sfreq) > 1e-6:
        raw.resample(args.sfreq)
    preprocess(raw)
    X = raw.get_data().astype(np.float64)
    N, B = args.n_components, args.max_iter
    Hs = list(range(1, args.h_max + 1))
    fitkw = dict(n_components=N, max_iter=B, num_mix=args.num_mix)
    print(f"[diag] {args.dataset} sub-{args.subject:02d} X={X.shape} N={N} B={B} seeds={args.seeds} n_surr={args.n_surr}")

    # ---- in-sample arm: real + surrogate H-sweeps (best-of-S) ----
    ll_real = {H: _insample_best_of_S(_fit_amica, X, H, args.seeds, **fitkw) for H in Hs}
    rng = np.random.default_rng(0)
    surrogates = [phase_randomize(X, rng) for _ in range(args.n_surr)]
    ll_surr = [{H: _insample_best_of_S(_fit_amica, Xs, H, args.seeds, **fitkw) for H in Hs} for Xs in surrogates]

    d_real = {H: delta_ll(ll_real)[H]["delta"] for H in Hs}
    d_surr = [{H: delta_ll(s)[H]["delta"] for H in Hs} for s in ll_surr]
    E = {H: d_real[H] - float(np.nanmean([s[H] for s in d_surr])) for H in Hs}
    sd_surr = {H: float(np.nanstd([s[H] for s in d_surr], ddof=1)) if args.n_surr > 1 else float("nan") for H in Hs}
    snr_E = {H: (E[H] / sd_surr[H] if sd_surr[H] and np.isfinite(sd_surr[H]) and sd_surr[H] > 0 else None) for H in Hs}
    surr2 = float(np.nanmean([s[2] for s in d_surr])) if 2 in Hs else float("nan")
    ratio_H2 = float(d_real[2] / surr2) if surr2 not in (0.0, float("nan")) and np.isfinite(surr2) and surr2 != 0 else None
    p95_surr2 = float(np.nanpercentile([s[2] for s in d_surr], 95)) if args.n_surr > 1 else float("nan")
    warranted = bool(E[2] > max(p95_surr2, 0.0)) if 2 in Hs else False
    knee_frac = {f"{f:g}": _knee_fraction(E, f) for f in args.knee_fracs}
    knee_chord = _knee_maxchord(E)

    # ---- held-out arm: noise floor of the per-step increment over (fold x seed) ----
    folds = block_folds(X.shape[1], args.heldout_folds)
    ho_incr = {}  # H -> list of (Lho(H)-Lho(H-1)) over fold x seed
    for H in args.heldout_h:
        vals = []
        for (tr, te) in folds:
            for s in args.seeds:
                try:
                    a = heldout_loglik(_fit_amica(X[:, tr], num_models=H, seed=int(s), **fitkw), X[:, te])
                    b = heldout_loglik(_fit_amica(X[:, tr], num_models=H - 1, seed=int(s), **fitkw), X[:, te])
                    vals.append(a - b)
                except Exception:
                    vals.append(np.nan)
        ho_incr[H] = vals
    ho_sd = {H: float(np.nanstd(v, ddof=1)) for H, v in ho_incr.items()}
    ho_mean = {H: float(np.nanmean(v)) for H, v in ho_incr.items()}
    ho_signal_over_sd = {H: (ho_mean[H] / ho_sd[H] if ho_sd[H] else None) for H in ho_incr}

    out = dict(
        dataset=args.dataset, subject=int(args.subject), n_channels=int(X.shape[0]),
        n_samples=int(X.shape[1]), n_components=N, max_iter=B, seeds=list(args.seeds), n_surr=args.n_surr,
        insample=dict(ll_real=ll_real, delta_real=d_real, delta_surr=d_surr, E=E, sd_surr=sd_surr,
                      SNR_E=snr_E, ratio_H2=ratio_H2, p95_surr2=p95_surr2, warranted=warranted,
                      knee_fraction=knee_frac, knee_maxchord=knee_chord),
        heldout=dict(increments=ho_incr, sd=ho_sd, mean=ho_mean, signal_over_sd=ho_signal_over_sd),
        config=vars(args))
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sanitize(out), indent=2), newline="\n")
    print(f"[diag] sub-{args.subject:02d} B={B}: warranted={warranted} SNR_E(2)={snr_E.get(2)} "
          f"ratio_H2={ratio_H2} knee={knee_frac} | held-out signal/sd={ho_signal_over_sd} -> {args.out}")


if __name__ == "__main__":
    main()
