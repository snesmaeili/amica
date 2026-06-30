"""
Shared pytest fixtures and backend-selection CLI option.

--backend=cpu   (default) Force JAX to use CPU
--backend=gpu           Force JAX to use GPU (skip if unavailable)
--backend=numpy         Bypass JAX entirely, use NumPy fallback

Markers:
  gpu   - skip unless --backend=gpu and a JAX GPU device is reachable
  slow  - skip unless --run-slow is passed
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ── CLI options ───────────────────────────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="cpu",
        choices=["cpu", "gpu", "numpy"],
        help="JAX backend to test against (default: cpu)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include slow tests (Fortran parity, long convergence)",
    )


def pytest_configure(config):
    backend = config.getoption("--backend", default="cpu")
    # Must be set before any jax import.
    if backend == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ.pop("AMICA_NO_JAX", None)
    elif backend == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ.pop("AMICA_NO_JAX", None)
    else:  # numpy
        os.environ["AMICA_NO_JAX"] = "1"

    config.addinivalue_line("markers", "gpu: requires --backend=gpu and a reachable JAX GPU")
    config.addinivalue_line("markers", "slow: long-running test; pass --run-slow to include")


def pytest_collection_modifyitems(config, items):
    backend = config.getoption("--backend", default="cpu")
    run_slow = config.getoption("--run-slow", default=False)

    skip_gpu = pytest.mark.skip(reason="requires --backend=gpu")
    skip_slow = pytest.mark.skip(reason="slow test; pass --run-slow to include")

    for item in items:
        if "gpu" in item.keywords and backend != "gpu":
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


# ── Session-scoped fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def tiny_data(rng):
    """4 channels x 200 samples — fast smoke-test data."""
    return rng.randn(4, 200).astype(np.float64)


@pytest.fixture(scope="session")
def laplacian_mix(rng):
    """4 independent Laplacian sources mixed through a random matrix."""
    n_src, n_samp = 4, 5000
    S = rng.laplace(size=(n_src, n_samp))
    A = rng.randn(n_src, n_src)
    return A @ S, A


@pytest.fixture(scope="session")
def synthetic_raw(rng):
    """8-channel 2000-sample MNE RawArray (skips if MNE not installed)."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    n_ch, n_samp = 8, 2000
    data = rng.randn(n_ch, n_samp) * 1e-6
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq=256.0, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture(scope="session")
def active_backend():
    """Return the string name of the active JAX backend (or 'numpy')."""
    try:
        import jax

        return jax.default_backend()
    except Exception:
        return "numpy"
