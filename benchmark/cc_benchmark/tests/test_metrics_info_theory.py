"""Unit tests for metrics_info_theory.py.

Run via:
    python -m pytest scripts/cc_benchmark/tests/test_metrics_info_theory.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


THIS_DIR = Path(__file__).resolve().parent
CC_BENCH = THIS_DIR.parent
if str(CC_BENCH) not in sys.path:
    sys.path.insert(0, str(CC_BENCH))

from metrics_info_theory import (  # noqa: E402
    compute_kappa,
    entropy_histogram_bits,
    pairwise_mi_matrix_bits,
    mean_offdiag_pairwise_mi_bits,
    complete_mir_bits_per_sample,
    remnant_pmi_percent,
    validate_square_mir_inputs,
)


# ---------------------------------------------------------------------------
# κ
# ---------------------------------------------------------------------------

def test_kappa_arithmetic():
    assert compute_kappa(150_000, 120) == pytest.approx(150_000 / (120 ** 2))
    assert compute_kappa(780_000, 120) == pytest.approx(780_000 / (120 ** 2))


def test_kappa_zero_channels_raises():
    with pytest.raises(ValueError):
        compute_kappa(1000, 0)


# ---------------------------------------------------------------------------
# Identity / permutation transforms -> MIR ~ 0
# ---------------------------------------------------------------------------

def _gaussian_sources(rng, n=4, t=20_000):
    return rng.standard_normal((n, t))


def test_identity_transform_zero_mir():
    rng = np.random.default_rng(0)
    X = _gaussian_sources(rng)
    Y = X.copy()
    W = np.eye(X.shape[0])
    result = complete_mir_bits_per_sample(X, Y, W, sfreq_hz=250.0, n_bins=100, max_samples=None)
    # MIR should be ~0 for identity. Histogram noise gives < ~0.5 bits/sample.
    assert abs(result.bits_per_sample) < 0.5, result


def test_permutation_and_sign_transform_zero_mir():
    rng = np.random.default_rng(1)
    X = _gaussian_sources(rng)
    perm = np.array([2, 0, 3, 1])
    signs = np.array([1, -1, 1, -1])
    W = (np.eye(4)[perm] * signs[:, None]).astype(float)
    Y = W @ X
    result = complete_mir_bits_per_sample(X, Y, W, sfreq_hz=250.0, n_bins=100, max_samples=None)
    assert abs(result.bits_per_sample) < 0.5, result


def test_orthogonal_rotation_zero_mir():
    rng = np.random.default_rng(2)
    X = _gaussian_sources(rng)
    # Orthogonal W has |det W| = 1; entropies unchanged.
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    Y = Q @ X
    result = complete_mir_bits_per_sample(X, Y, Q, sfreq_hz=250.0, n_bins=100, max_samples=None)
    assert abs(result.bits_per_sample) < 0.5, result


# ---------------------------------------------------------------------------
# PMI matrix properties
# ---------------------------------------------------------------------------

def test_pmi_diagonal_is_zero():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 20_000))
    pmi = pairwise_mi_matrix_bits(X, n_bins=50, max_samples=None)
    assert np.allclose(np.diag(pmi), 0)


def test_pmi_is_symmetric():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4, 10_000))
    pmi = pairwise_mi_matrix_bits(X, n_bins=50, max_samples=None)
    assert np.allclose(pmi, pmi.T)


def test_pmi_nonneg_for_independent_gaussians():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 20_000))
    mean = mean_offdiag_pairwise_mi_bits(X, n_bins=50, max_samples=None)
    # Independent gaussians -> MI ~ 0 but histogram bias makes it small-positive.
    assert mean >= 0
    assert mean < 0.5


# ---------------------------------------------------------------------------
# remnant PMI
# ---------------------------------------------------------------------------

def test_remnant_pmi_nonneg():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((4, 10_000))
    Y = rng.standard_normal((4, 10_000))
    d = remnant_pmi_percent(X, Y, n_bins=50, max_samples=None)
    assert d["remnant_pmi_percent"] >= 0


# ---------------------------------------------------------------------------
# Rectangular W requires subspace_mode
# ---------------------------------------------------------------------------

def test_rectangular_W_requires_subspace_mode():
    X = np.random.randn(5, 1000)
    Y = np.random.randn(3, 1000)
    W = np.random.randn(3, 5)        # rectangular: PCA-then-ICA without subspace flag
    with pytest.raises(ValueError, match="not square"):
        validate_square_mir_inputs(X, Y, W, subspace_mode=False)
    # subspace_mode=True still raises in validator because shape is not square,
    # but if you pass a square W of the retained-rank space, it should pass:
    W_sq = np.random.randn(3, 3)
    Y_sq = np.random.randn(3, 1000)
    X_sq = np.random.randn(3, 1000)
    validate_square_mir_inputs(X_sq, Y_sq, W_sq, subspace_mode=True)


def test_mismatched_samples_raises():
    X = np.random.randn(3, 1000)
    Y = np.random.randn(3, 999)
    with pytest.raises(ValueError, match="same number of samples"):
        validate_square_mir_inputs(X, Y, None, subspace_mode=False)


# ---------------------------------------------------------------------------
# Singular W -> error
# ---------------------------------------------------------------------------

def test_singular_W_raises():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 5_000))
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # singular
    Y = W @ X
    with pytest.raises(ValueError, match="singular"):
        complete_mir_bits_per_sample(X, Y, W, sfreq_hz=250.0, n_bins=50, max_samples=None)
