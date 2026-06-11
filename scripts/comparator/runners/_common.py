"""Shared runner protocol for the three-implementation perf comparison.

Each runner script:
  - Takes --input (path to .npz with key 'X' = (n_components, n_samples)).
  - Takes --output (path to write the result JSON).
  - Takes --config (JSON string with at minimum: max_iter, n_mix, lrate, seed).
  - Writes a JSON dict with the keys defined in `RESULT_KEYS` below.

Peak-RSS is the process high-water mark: resource.getrusage(RUSAGE_SELF).ru_maxrss
on POSIX (the TRUE peak — captures a transient high that occurred mid-fit, not just
the instantaneous RSS at call time) and psutil peak_wset on Windows. Each runner
records a pre-fit baseline (baseline_rss_gb) right before fit() and reports
delta_rss_gb = peak - baseline (the fit's marginal footprint) next to the absolute
peak. peak_vram_gb is the GPU device peak when on GPU (jax peak_bytes_in_use /
torch max_memory_allocated), else None. All values are GiB (binary, /1024**n) to
match the in-tree benchmark.runner convention.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

RESULT_KEYS = (
    "implementation", "n_components", "n_samples", "max_iter",
    "fit_time_s", "peak_rss_gb", "baseline_rss_gb", "delta_rss_gb",
    "peak_vram_gb", "nvml_peak_vram_gb",
    "ll_final", "ll_history", "W",
    "device", "dtype", "n_iter",
)


def parse_runner_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npz with key X (n_components, n_samples)")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--config", required=True, help="JSON-encoded config dict")
    args = parser.parse_args()
    cfg = json.loads(args.config)
    return args, cfg


def load_data(input_path: str) -> np.ndarray:
    z = np.load(input_path)
    X = z["X"].astype(np.float64)
    return X  # (n_components, n_samples)


def peak_rss_gb() -> float:
    """Process peak resident-set size (high-water mark), in GiB.

    POSIX: resource.getrusage(RUSAGE_SELF).ru_maxrss — the kernel's TRUE
    high-water mark (KiB on Linux, bytes on macOS), so a transient peak that
    occurred mid-fit is captured, unlike the instantaneous psutil rss the old
    code returned on Linux. Windows: psutil peak_wset (also a high-water mark).
    Binary GiB (/1024**n) to match amica_python.benchmark.runner._measure_peak_memory.
    """
    if sys.platform != "win32":
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            div = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
            return float(ru) / div
        except Exception:
            pass
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows high-water mark (bytes)
        return info.peak_wset / 1024 ** 3
    return info.rss / 1024 ** 3


def baseline_rss_gb() -> float:
    """Pre-fit RSS baseline (GiB) — same high-water source as peak_rss_gb().

    Call right before fit() so the value is the post-import / pre-fit floor that
    delta_rss_gb() subtracts off. (peak_rss_gb is monotonic, so this is the
    high-water-so-far = interpreter + data + framework import.)
    """
    return peak_rss_gb()


def delta_rss_gb(baseline: float) -> float:
    """Peak RSS attributable to the fit: current peak minus the pre-fit baseline (GiB)."""
    return peak_rss_gb() - baseline


def start_nvml_sampler(enabled: bool, gpu_index: int = 0, interval_s: float = 0.05):
    """Start a background sampler of whole-GPU 'used' VRAM via NVML; return a handle.

    Framework-neutral cross-check: on a DEDICATED GPU (our `--gres=gpu:h100:1` case)
    the whole-GPU used peak = our process's total device footprint = allocator tensors
    (peak_bytes_in_use / max_memory_allocated) PLUS the fixed CUDA-context floor
    (cuDNN/cuBLAS, few-hundred-MB) that the per-framework allocator counters omit.
    Returns None if disabled or pynvml/GPU unavailable (caller treats as no cross-check).
    Pass the handle to stop_nvml_sampler() to read the peak. Requires `pip install
    nvidia-ml-py` in the venv; silently degrades to None otherwise.
    """
    if not enabled:
        return None
    try:
        import threading
        import pynvml
        pynvml.nvmlInit()
        handle_dev = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        state = {"stop": threading.Event(), "peak": 0.0, "nvml": pynvml, "thread": None}

        def _loop():
            while not state["stop"].is_set():
                try:
                    used = pynvml.nvmlDeviceGetMemoryInfo(handle_dev).used
                    g = float(used) / 1024 ** 3
                    if g > state["peak"]:
                        state["peak"] = g
                except Exception:
                    pass
                state["stop"].wait(interval_s)

        state["thread"] = threading.Thread(target=_loop, daemon=True)
        state["thread"].start()
        return state
    except Exception:
        return None


def stop_nvml_sampler(handle) -> float | None:
    """Stop the sampler and return peak whole-GPU used VRAM in GiB (or None)."""
    if not handle:
        return None
    try:
        handle["stop"].set()
        handle["thread"].join(timeout=1.0)
        handle["nvml"].nvmlShutdown()
        return float(handle["peak"])
    except Exception:
        return None


def write_result(output_path: str, result: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    if "error" in result:
        print(f"[{result.get('implementation', '?')}] ERROR: {result['error']}  -> {output_path}")
        return
    vram = result.get("peak_vram_gb")
    vram_s = f"  vram={vram:.2f}GB" if vram is not None else ""
    print(f"[{result['implementation']}] {result['fit_time_s']:.2f}s  "
          f"peak={result['peak_rss_gb']:.2f}GB  "
          f"delta={result.get('delta_rss_gb', float('nan')):.2f}GB{vram_s}  "
          f"ll={result['ll_final']:.4f}  -> {output_path}")
