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
from _common import (
    baseline_rss_gb,
    load_data,
    parse_runner_args,
    peak_rss_gb,
    start_nvml_sampler,
    stop_nvml_sampler,
    write_result,
)


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape

    import torch
    from pyamica import AMICA

    device = os.environ.get("TORCH_DEVICE", "cpu")
    Xt = torch.from_numpy(X.T).to(device)  # (n_samples, n_components)
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
        device=device,
        fix_init=True,
    )

    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    _nvml = start_nvml_sampler(_use_nvml)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    model.fit(Xt)
    elapsed = time.perf_counter() - t0

    # Torch device peak: bytes in live tensors (max_memory_allocated = true demand, NOT the
    # cached/reserved pool). The caching allocator stays ON so this counter is tracked.
    peak_vram_gb = None
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    W = model.W_[0].cpu().numpy()
    ll = model.LL_.cpu().numpy().tolist()

    peak = peak_rss_gb()
    out = {
        "implementation": "pyamica_torch",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak,
        "baseline_rss_gb": baseline,
        "delta_rss_gb": peak - baseline,
        "peak_vram_gb": peak_vram_gb,
        "nvml_peak_vram_gb": nvml_peak_vram_gb,
        "ll_final": float(ll[-1]) if ll else float("nan"),
        "ll_history": ll,
        "W": W.tolist(),
        "device": device,
        "dtype": "float64",
        "n_iter": int(len(ll)),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
