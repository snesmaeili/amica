#!/usr/bin/env python
"""Held-out / cross-validated MIR (reviewer Major 4.1).

The headline real-EEG MIR result is in-sample (fit and scored on the same
samples). This driver converts it to an out-of-sample quantity: split the
recording into K contiguous time blocks, fit each method on the training
blocks, and evaluate MIR on the held-out block using the train-fitted PCA +
unmixing (``complete_mir_from_ica`` projects the test data through the fitted
model). Held-out MIR is comparable across all four methods; the in-sample MIR
is also recorded per fold so the optimism gap is visible.

Run (smoke):
  python run_heldout_cv.py --dataset mne --subject 0 --n-components 20 \
      --folds 3 --max-iter 60 --out /tmp/heldout_mne.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

COMPARATORS = ("picard", "infomax", "fastica")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="mne | ds004505 | ds004504 | ds004621")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=2000, help="comparator ceiling")
    ap.add_argument("--amica-max-iter", type=int, default=None, help="defaults to --max-iter")
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"],
                    help="device for AMICA's JAX backend (comparators always run on CPU)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Route AMICA's JAX backend exactly as the main runner does: set the platform
    # BEFORE importing amica_python so the backend picks it up.
    os.environ.setdefault("AMICA_NO_JAX", "0")
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if args.device == "gpu" else "cpu"

    import mne
    mne.set_log_level("ERROR")
    from amica_python import fit_ica
    from amica_python.benchmark.runner import load_data, preprocess, DATASET_LINE_FREQ, DATASET_RESAMPLE
    from amica_python.benchmark.comparators import fit_mne_ica
    from amica_python.benchmark.metrics import complete_mir_from_ica

    # Load + preprocess identically to the main benchmark (FIR 1-100 Hz, per-site
    # notch, standardise to 250 Hz where applicable), then work in EEG sensor space.
    raw = load_data(args.dataset, args.subject)
    raw.load_data()
    preprocess(raw, line_freq=DATASET_LINE_FREQ.get(args.dataset, 60.0))
    rs = DATASET_RESAMPLE.get(args.dataset)
    if rs:
        raw.resample(rs)
    raw.pick("eeg")

    data = raw.get_data()
    info = raw.info
    n_ch, T = data.shape
    K = int(args.folds)
    bounds = np.linspace(0, T, K + 1).astype(int)
    amica_iter = int(args.amica_max_iter or args.max_iter)

    def score(test_raw, train_raw, ica):
        return (float(complete_mir_from_ica(test_raw, ica).kbits_per_sec),
                float(complete_mir_from_ica(train_raw, ica).kbits_per_sec))

    results = []
    for k in range(K):
        a, b = int(bounds[k]), int(bounds[k + 1])
        test_raw = mne.io.RawArray(data[:, a:b], info, verbose="ERROR")
        train_raw = mne.io.RawArray(np.concatenate([data[:, :a], data[:, b:]], axis=1), info, verbose="ERROR")

        t0 = time.perf_counter()
        ica = fit_ica(train_raw, n_components=args.n_components, max_iter=amica_iter,
                      num_mix=args.num_mix, random_state=args.random_state)
        dt = time.perf_counter() - t0
        ho, ins = score(test_raw, train_raw, ica)
        results.append(dict(fold=k, method="AMICA-Python", heldout_mir_kbits_s=ho,
                            insample_mir_kbits_s=ins, fit_s=dt))
        print(f"fold {k} AMICA-Python   heldout={ho:.4f}  insample={ins:.4f}  ({dt:.1f}s)")

        for m in COMPARATORS:
            t0 = time.perf_counter()
            ica_c, elapsed, _ = fit_mne_ica(train_raw, m, args.n_components,
                                            args.random_state, max_iter=args.max_iter)
            ho, ins = score(test_raw, train_raw, ica_c)
            results.append(dict(fold=k, method=m, heldout_mir_kbits_s=ho,
                                insample_mir_kbits_s=ins, fit_s=float(elapsed)))
            print(f"fold {k} {m:<12} heldout={ho:.4f}  insample={ins:.4f}  ({elapsed:.1f}s)")

    payload = dict(dataset=args.dataset, subject=args.subject, n_components=args.n_components,
                   folds=K, sfreq=float(info["sfreq"]), n_channels=n_ch, n_times=int(T),
                   amica_max_iter=amica_iter, comparator_max_iter=int(args.max_iter),
                   random_state=args.random_state, results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
