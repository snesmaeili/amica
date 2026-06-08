"""Runner for DerAndereJohannes/pyamica (PyTorch).

Pyamica's AMICA accepts (T, n_components) tensors and applies its own
sphering when do_sphere=True. Our orchestrator pre-PCA-projects, so we
pass do_sphere=False and feed already-whitened data.
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
    import torch
    from pyamica import AMICA

    Xt = torch.from_numpy(X.T)  # (n_samples, n_components)
    model = AMICA(
        n_components=n_comp,
        n_models=1,
        n_mix=cfg.get("n_mix", 3),
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        lrate0=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        newt_start=50,
        newt_ramp=10,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
        rholrate=0.05,
        invsigmin=1e-8,
        invsigmax=100.0,
        do_sphere=False,
        doscaling=True,
        verbose=False,
        dtype=torch.float64,
        device="cpu",
        fix_init=True,
    )

    t0 = time.perf_counter()
    model.fit(Xt)
    elapsed = time.perf_counter() - t0

    W = model.W_[0].cpu().numpy()
    ll = model.LL_.cpu().numpy().tolist()

    out = {
        "implementation": "pyamica_torch",
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
