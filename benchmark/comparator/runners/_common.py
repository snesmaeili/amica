"""Shared runner protocol for the three-implementation perf comparison.

Each runner script:
  - Takes --input (path to .npz with key 'X' = (n_components, n_samples)).
  - Takes --output (path to write the result JSON).
  - Takes --config (JSON string with at minimum: max_iter, n_mix, lrate, seed).
  - Writes a JSON dict with the keys defined in `RESULT_KEYS` below.

Peak-RSS measurement uses psutil.Process().memory_info().peak_wset on
Windows and ru_maxrss on Linux. The runner samples *before* importing
the implementation under test so the baseline is the venv-only RSS.
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
    "fit_time_s", "peak_rss_gb", "ll_final", "ll_history", "W",
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
    """Cross-platform peak resident-set size, in GB."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 1e9
    if hasattr(info, "ru_maxrss"):  # POSIX (KB on Linux, B on macOS)
        rss = info.ru_maxrss
        if sys.platform == "darwin":
            return rss / 1e9
        return rss * 1024 / 1e9
    return info.rss / 1e9


def write_result(output_path: str, result: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{result['implementation']}] {result['fit_time_s']:.2f}s  "
          f"peak={result['peak_rss_gb']:.2f}GB  ll={result['ll_final']:.4f}  "
          f"-> {output_path}")
