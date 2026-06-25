"""SCHL pilot driver — run data-driven AMICA configuration on one real-EEG subject.

Loads a subject through the project's standard loader+preprocess, runs
``amica_python.selector.auto_select_amica`` (surrogate-calibrated held-out likelihood),
and writes the selected (N*, M*, rejsig*) + the full criterion curves to a JSON.

This is the Stage-1 *pilot* driver (de-risk the criterion + pin n_surr / iter budget on a
few subjects) BEFORE the full Stage-2 grid. CPU is the right home: the selector fits many
small AMICA models, which are not GPU-bound.

Example (one subject):
    python run_auto_select.py --dataset ds004505 --subject 1 \
        --n-components-grid 32 64 --h-max 6 --rejsig-grid 3.0 2.5 \
        --k-folds 3 --n-surr 10 --max-iter 600 --out out/schl_ds004505_sub01.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _sanitize(obj):
    """Recursively make a dict JSON-safe (NaN/Inf -> None, numpy scalars -> python)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds004505")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--input-level", default="bids")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--n-components-grid", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--h-max", type=int, default=6)
    ap.add_argument("--rejsig-grid", type=float, nargs="+", default=[3.0, 2.5],
                    help="rejsig candidates; 'off' (None) is prepended automatically")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--n-surr", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--kappa-min", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from amica_python.benchmark.runner import load_data, preprocess
    from amica_python.selector import auto_select_amica

    raw = load_data(args.dataset, args.subject, input_level=args.input_level)
    raw.load_data()
    if args.sfreq and abs(float(raw.info["sfreq"]) - args.sfreq) > 1e-6:
        raw.resample(args.sfreq)
    preprocess(raw)                                   # project-standard band-pass + notch
    X = raw.get_data().astype(np.float64)             # (n_channels, n_samples)
    print(f"[schl] {args.dataset} sub-{args.subject:02d}: X={X.shape} sfreq={raw.info['sfreq']:.1f}")

    rejsig_grid = tuple([None] + list(args.rejsig_grid))
    rep = auto_select_amica(
        X, N_grid=list(args.n_components_grid), H_max=args.h_max, rejsig_grid=rejsig_grid,
        k_folds=args.k_folds, n_surr=args.n_surr, max_iter=args.max_iter,
        num_mix=args.num_mix, kappa_min=args.kappa_min, seed=args.seed,
        rng=np.random.default_rng(args.seed))

    out = dict(
        dataset=args.dataset, subject=int(args.subject),
        n_channels=int(X.shape[0]), n_samples=int(X.shape[1]), sfreq=float(raw.info["sfreq"]),
        selected=dict(n_components=rep.n_components, num_models=rep.num_models,
                      rejsig=rep.rejsig, do_reject=rep.do_reject,
                      kappa_channels=rep.kappa_channels, kappa_effective=rep.kappa_effective),
        rank_info=rep.rank_info, model_order_info=rep.model_order_info,
        rejection_info=rep.rejection_info, config=vars(args))
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_sanitize(out), indent=2), newline="\n")
    print(f"[schl] sub-{args.subject:02d}: N*={rep.n_components} M*={rep.num_models} "
          f"rejsig*={rep.rejsig} -> {args.out}")


if __name__ == "__main__":
    main()
