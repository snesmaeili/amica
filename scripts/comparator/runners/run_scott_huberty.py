"""Runner for scott-huberty/amica-python (PyTorch, sklearn-style).

Scott's `amica.AMICA` is sklearn-compatible: fit(X) where X is
(n_samples, n_features). We pass already-PCA-projected data and disable
its internal whitening via whiten=None / batching=None.
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
    from amica import AMICA  # Scott's sklearn-style class

    device = os.environ.get("TORCH_DEVICE", "cpu")
    # sklearn fits on (n_samples, n_features); transpose
    Xt = X.T

    model = AMICA(
        n_components=n_comp,
        n_mixtures=cfg.get("n_mix", 3),
        device=device,
        n_models=1,
        mean_center=False,
        whiten=None,                     # already projected
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        newt_start=50,
        random_state=cfg.get("seed", 0),
        verbose=0,
    )

    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    _nvml = start_nvml_sampler(_use_nvml)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    model.fit(Xt)
    elapsed = time.perf_counter() - t0

    # Torch device peak: bytes in live tensors (max_memory_allocated, NOT the
    # cached reserve), captured with PYTORCH_NO_CUDA_MEMORY_CACHING=1 by the orchestrator.
    peak_vram_gb = None
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    # Scott's sklearn-style attributes: components_ (unmixing), ll_ (per-iter), n_iter_
    W = np.asarray(model.components_)
    if W.ndim == 3:
        W = W[0]
    ll = np.asarray(model.ll_).flatten().tolist()
    n_iter = int(model.n_iter_)

    peak = peak_rss_gb()
    out = {
        "implementation": "scott_huberty_torch",
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
        "W": W.tolist() if W is not None else None,
        "device": device,
        "dtype": str(np.asarray(W).dtype) if W is not None else "float64",
        "n_iter": n_iter,
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
