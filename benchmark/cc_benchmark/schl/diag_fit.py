"""SCHL diagnostic fan-out WORKER: compute ONE cell of fits and write its LLs.

The diagnostic's fits are independent, so we split them across array tasks (many GPUs in
parallel) instead of ~170 sequential fits per subject. Each invocation does one `--job`:
  real            -> in-sample best-of-S LL per H on the recording        -> {H: ll}
  surr_<J>        -> same on phase-surrogate J (deterministic seed)        -> {H: ll}
  ho_<F>          -> held-out per-step increment on CV fold F (the noise)  -> [incr per seed]
Reduce locally with diag_reduce.py. Mirrors run_diagnostic.py's math, just sliced.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _san(o):
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o); return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds004505")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--input-level", default="bids")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--h-max", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-iter", type=int, required=True)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--heldout-folds", type=int, default=5)
    ap.add_argument("--heldout-h", type=int, default=2)
    ap.add_argument("--job", required=True, help="real | surr_<J> | ho_<F>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from amica_python.benchmark.runner import load_data, preprocess
    from amica_python.selector import _fit_amica, heldout_loglik, phase_randomize, block_folds

    raw = load_data(a.dataset, a.subject, input_level=a.input_level)
    raw.load_data()
    if a.sfreq and abs(float(raw.info["sfreq"]) - a.sfreq) > 1e-6:
        raw.resample(a.sfreq)
    preprocess(raw)
    X = raw.get_data().astype(np.float64)
    Hs = list(range(1, a.h_max + 1))
    kw = dict(n_components=a.n_components, max_iter=a.max_iter, num_mix=a.num_mix)

    def best_of_S(Xd, H):
        v = []
        for s in a.seeds:
            try:
                v.append(float(_fit_amica(Xd, num_models=H, seed=int(s), **kw).log_likelihood[-1]))
            except Exception:
                v.append(np.nan)
        return float(np.nanmax(v)) if np.any(np.isfinite(v)) else np.nan

    rec = dict(job=a.job, subject=int(a.subject), max_iter=int(a.max_iter), n_components=int(a.n_components),
               n_samples=int(X.shape[1]), n_channels=int(X.shape[0]), seeds=list(a.seeds))
    if a.job == "real":
        rec.update(kind="insample", ll={int(H): best_of_S(X, H) for H in Hs})
    elif a.job.startswith("surr_"):
        J = int(a.job.split("_")[1])
        Xs = phase_randomize(X, np.random.default_rng(1000 + J))   # deterministic surrogate J
        rec.update(kind="insample_surr", surr_idx=J, ll={int(H): best_of_S(Xs, H) for H in Hs})
    elif a.job.startswith("ho_"):
        F = int(a.job.split("_")[1])
        tr, te = block_folds(X.shape[1], a.heldout_folds)[F]
        incr = []
        for s in a.seeds:
            try:
                hi = heldout_loglik(_fit_amica(X[:, tr], num_models=a.heldout_h, seed=int(s), **kw), X[:, te])
                lo = heldout_loglik(_fit_amica(X[:, tr], num_models=a.heldout_h - 1, seed=int(s), **kw), X[:, te])
                incr.append(float(hi - lo))
            except Exception:
                incr.append(np.nan)
        rec.update(kind="heldout", fold=F, heldout_h=a.heldout_h, increments=incr)
    else:
        raise SystemExit(f"bad --job {a.job!r} (real | surr_<J> | ho_<F>)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_san(rec), indent=2), newline="\n")
    print(f"[diagfit] sub-{a.subject:02d} B={a.max_iter} {a.job} -> {a.out}")


if __name__ == "__main__":
    main()
