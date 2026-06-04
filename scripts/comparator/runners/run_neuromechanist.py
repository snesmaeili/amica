"""Runner for neuromechanist/pyAMICA (pure NumPy).

The orchestrator pre-PCA-projects data, so we pass do_sphere=False and
do_mean=False. fit(data) expects (n_channels, n_samples), which matches
the orchestrator's (n_components, n_samples) layout — no transpose.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_data, parse_runner_args, peak_rss_gb, write_result


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape

    _ = peak_rss_gb()
    from pyAMICA import AMICA  # neuromechanist's class

    model = AMICA(
        num_models=1,
        num_mix=cfg.get("n_mix", 3),
        num_comps=n_comp,
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        newt_start=50,
        do_sphere=False,            # already PCA-projected by orchestrator
        do_mean=False,
        do_opt_block=False,         # skip block-size optimization (would steal wall time)
        do_history=False,
        do_reject=False,
        share_comps=False,
        seed=cfg.get("seed", 0),
        verbose=False,
        use_tqdm=False,
    )

    t0 = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - t0

    # model.W has shape (n_components, n_components, n_models); take first model
    W = np.asarray(model.W[:, :, 0], dtype=float)
    ll = list(np.asarray(model.ll, dtype=float).flatten())

    out = {
        "implementation": "neuromechanist_numpy",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak_rss_gb(),
        "ll_final": float(ll[-1]) if ll else float("nan"),
        "ll_history": ll,
        "W": W.tolist(),
        "device": "cpu",
        "dtype": "float64",
        "n_iter": int(len(ll)),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
