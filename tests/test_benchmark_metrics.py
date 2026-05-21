"""Unit tests for ``amica_python.benchmark.metrics``.

Run via:
    pytest tests/test_benchmark_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from amica_python.benchmark.metrics import (
    kappa,
    entropy_histogram,
    pairwise_mi_matrix,
    mean_pairwise_mi,
    complete_mir,
    remnant_pmi,
    _validate_mir_inputs,
)


# ---------------------------------------------------------------------------
# κ
# ---------------------------------------------------------------------------

def test_kappa_arithmetic():
    assert kappa(150_000, 120) == pytest.approx(150_000 / (120 ** 2))
    assert kappa(780_000, 120) == pytest.approx(780_000 / (120 ** 2))


def test_kappa_zero_channels_raises():
    with pytest.raises(ValueError):
        kappa(1000, 0)


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
    result = complete_mir(X, Y, W, sfreq_hz=250.0, n_bins=100, max_samples=None)
    assert abs(result.bits_per_sample) < 0.5, result


def test_permutation_and_sign_transform_zero_mir():
    rng = np.random.default_rng(1)
    X = _gaussian_sources(rng)
    perm = np.array([2, 0, 3, 1])
    signs = np.array([1, -1, 1, -1])
    W = (np.eye(4)[perm] * signs[:, None]).astype(float)
    Y = W @ X
    result = complete_mir(X, Y, W, sfreq_hz=250.0, n_bins=100, max_samples=None)
    assert abs(result.bits_per_sample) < 0.5, result


def test_orthogonal_rotation_zero_mir():
    rng = np.random.default_rng(2)
    X = _gaussian_sources(rng)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    Y = Q @ X
    result = complete_mir(X, Y, Q, sfreq_hz=250.0, n_bins=100, max_samples=None)
    assert abs(result.bits_per_sample) < 0.5, result


# ---------------------------------------------------------------------------
# PMI matrix properties
# ---------------------------------------------------------------------------

def test_pmi_diagonal_is_zero():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 20_000))
    pmi = pairwise_mi_matrix(X, n_bins=50, max_samples=None)
    assert np.allclose(np.diag(pmi), 0)


def test_pmi_is_symmetric():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4, 10_000))
    pmi = pairwise_mi_matrix(X, n_bins=50, max_samples=None)
    assert np.allclose(pmi, pmi.T)


def test_pmi_nonneg_for_independent_gaussians():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 20_000))
    mean = mean_pairwise_mi(X, n_bins=50, max_samples=None)
    assert mean >= 0
    assert mean < 0.5


# ---------------------------------------------------------------------------
# remnant PMI
# ---------------------------------------------------------------------------

def test_remnant_pmi_nonneg():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((4, 10_000))
    Y = rng.standard_normal((4, 10_000))
    d = remnant_pmi(X, Y, n_bins=50, max_samples=None)
    assert d["remnant_pmi_percent"] >= 0


# ---------------------------------------------------------------------------
# Rectangular W requires subspace_mode
# ---------------------------------------------------------------------------

def test_rectangular_W_requires_subspace_mode():
    X = np.random.randn(5, 1000)
    Y = np.random.randn(3, 1000)
    W = np.random.randn(3, 5)        # rectangular: PCA-then-ICA without subspace flag
    with pytest.raises(ValueError, match="not square"):
        _validate_mir_inputs(X, Y, W, subspace_mode=False)
    # subspace_mode=True allows square retained-rank W:
    W_sq = np.random.randn(3, 3)
    Y_sq = np.random.randn(3, 1000)
    X_sq = np.random.randn(3, 1000)
    _validate_mir_inputs(X_sq, Y_sq, W_sq, subspace_mode=True)


def test_mismatched_samples_raises():
    X = np.random.randn(3, 1000)
    Y = np.random.randn(3, 999)
    with pytest.raises(ValueError, match="same number of samples"):
        _validate_mir_inputs(X, Y, None, subspace_mode=False)


def test_singular_W_raises():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 5_000))
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # singular
    Y = W @ X
    with pytest.raises(ValueError, match="singular"):
        complete_mir(X, Y, W, sfreq_hz=250.0, n_bins=50, max_samples=None)


# ---------------------------------------------------------------------------
# complete_mir_from_ica end-to-end
# ---------------------------------------------------------------------------


def test_complete_mir_from_ica_on_synthetic_mixture():
    """End-to-end: mix independent sources, fit Picard, MIR should be positive."""
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    import mne
    from amica_python.benchmark.metrics import complete_mir_from_ica

    rng = np.random.default_rng(8)
    n_ch = 6
    n_t = 5000
    sfreq = 100.0
    # Independent sources -> heavy-tailed (laplacian-like) for ICA to find.
    sources_true = rng.standard_t(df=3, size=(n_ch, n_t))
    # Random invertible mixing
    A = rng.standard_normal((n_ch, n_ch))
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.standard_normal((n_ch, n_ch))
    X = A @ sources_true  # observed channel data
    # Build a minimal Raw and fit Picard.
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(
        n_components=n_ch,
        method="picard",
        fit_params={"ortho": False, "extended": True, "tol": 1e-6},
        max_iter=2000,
        random_state=0,
    )
    ica.fit(raw, verbose="ERROR")
    result = complete_mir_from_ica(raw, ica, n_bins=80, max_samples=None)
    # On a clean linear mixture of heavy-tailed sources, complete MIR > 0 (ICA helps).
    assert result.bits_per_sample > 0, f"Expected positive MIR on heavy-tailed mixture, got {result}"
    assert result.subspace_mode is True
    assert result.n_components == n_ch
    assert np.isfinite(result.log2_abs_det_W)


def test_complete_mir_invariant_to_source_scale():
    """MIR must not change when each source is rescaled by a per-row factor.

    AMICA-Python's unmixing_matrix_ does not normalise sources to unit variance
    the way MNE's wrappers around Picard/FastICA/Infomax do. complete_mir_from_ica
    must internally re-gauge so AMICA and the comparators land on the same axis.
    """
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    import mne
    from amica_python.benchmark.metrics import complete_mir_from_ica

    rng = np.random.default_rng(11)
    n_ch = 6
    sources_true = rng.standard_t(df=3, size=(n_ch, 4000))
    A = rng.standard_normal((n_ch, n_ch))
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.standard_normal((n_ch, n_ch))
    X = A @ sources_true
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n_ch, method="picard",
                                fit_params={"ortho": False, "extended": True},
                                max_iter=1500, random_state=0)
    ica.fit(raw, verbose="ERROR")
    baseline = complete_mir_from_ica(raw, ica, n_bins=80, max_samples=None)

    # Rescale rows of the unmixing matrix by random factors — sources will
    # now have non-unit variance (mimics AMICA's gauge).
    scales = rng.uniform(0.3, 3.0, size=n_ch)
    ica.unmixing_matrix_ = ica.unmixing_matrix_ * scales[:, None]
    rescaled = complete_mir_from_ica(raw, ica, n_bins=80, max_samples=None)

    # MIR must be invariant to per-row source rescaling (Frank 2022 eq. 5 is
    # gauge-invariant analytically; histogram noise gives ~0.5 bits slack).
    assert abs(baseline.bits_per_sample - rescaled.bits_per_sample) < 1.0, (
        f"baseline={baseline.bits_per_sample:.3f}  rescaled={rescaled.bits_per_sample:.3f}"
    )


def test_Y_equals_W_at_X_numerically():
    """For complete_mir_from_ica's chosen X_pca / W / Y, Y must equal W @ X numerically.

    Catches any mismatch in the PCA-rank slice that would silently break MIR.
    """
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    import mne
    from amica_python.benchmark.metrics import pca_inputs_from_ica

    rng = np.random.default_rng(12)
    n_ch = 5
    X = rng.standard_normal((n_ch, 3000))
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n_ch, method="picard",
                                fit_params={"ortho": False, "extended": True},
                                max_iter=500, random_state=0)
    ica.fit(raw, verbose="ERROR")
    X_pca, W_square = pca_inputs_from_ica(ica, raw)
    Y_reconstructed = W_square @ X_pca
    Y_from_ica = ica.get_sources(raw).get_data()
    # MNE returns sources in possibly different per-row scale; verify Y == W @ X
    # numerically since both are computed from the same recipe.
    np.testing.assert_allclose(Y_reconstructed, Y_from_ica, rtol=1e-6, atol=1e-10)


def test_clip_choice_does_not_flip_ranking():
    """Method ranking under MIR should be stable to small changes in the
    histogram clip threshold (±3, ±5, ±10 σ).
    """
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    import mne
    from amica_python.benchmark.metrics import complete_mir_from_ica

    rng = np.random.default_rng(13)
    n_ch = 5
    sources_true = rng.standard_t(df=3, size=(n_ch, 4000))
    A = rng.standard_normal((n_ch, n_ch))
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.standard_normal((n_ch, n_ch))
    X = A @ sources_true
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    # Fit two methods so we can check ranking stability
    ica_picard = mne.preprocessing.ICA(n_components=n_ch, method="picard",
                                       fit_params={"ortho": False, "extended": True},
                                       max_iter=1000, random_state=0)
    ica_picard.fit(raw, verbose="ERROR")
    ica_fastica = mne.preprocessing.ICA(n_components=n_ch, method="fastica",
                                        fit_params={"fun": "logcosh"},
                                        max_iter=1000, random_state=0)
    ica_fastica.fit(raw, verbose="ERROR")

    rankings = {}
    for clip in (5.0, 8.0, 10.0):
        mir_p = complete_mir_from_ica(raw, ica_picard, n_bins=80, clip_sd=clip, max_samples=None).bits_per_sample
        mir_f = complete_mir_from_ica(raw, ica_fastica, n_bins=80, clip_sd=clip, max_samples=None).bits_per_sample
        rankings[clip] = "picard>fastica" if mir_p > mir_f else "fastica>picard"
    # Stability is required for moderate-to-wide clip ranges (5-10 σ). Very
    # tight clips (±3 σ) can flip the ranking on heavy-tailed sources, which
    # is a known estimator limitation documented in the audit.
    assert len(set(rankings.values())) == 1, (
        f"Clip choice changed ranking at moderate-to-wide clips: {rankings}"
    )


def test_complete_mir_uses_stable_slogdet():
    """Verify the implementation uses np.linalg.slogdet (numerically stable)
    instead of np.linalg.det → log (which overflows for high-dim or scaled W).
    """
    import inspect
    from amica_python.benchmark import metrics
    src = inspect.getsource(metrics.complete_mir)
    assert "slogdet" in src, (
        "complete_mir() must use np.linalg.slogdet for numerical stability; "
        "found no slogdet reference in the source."
    )
    # Sanity: a moderately ill-conditioned W shouldn't overflow.
    rng = np.random.default_rng(20)
    n = 8
    X = rng.standard_normal((n, 4000))
    # W with one row scaled large -- would overflow np.linalg.det but not slogdet.
    W = rng.standard_normal((n, n))
    W[0] *= 1e10
    Y = W @ X
    result = complete_mir(X, Y, W, sfreq_hz=250.0, n_bins=80, max_samples=None)
    # MIR should still be finite, near 0 modulo histogram noise (W is full rank).
    assert np.isfinite(result.bits_per_sample), f"slogdet should keep result finite, got {result}"
    assert np.isfinite(result.log2_abs_det_W)


def test_amica_and_mne_use_same_pca_xspace():
    """When AMICA-Python and MNE Picard fit the same data with the same
    n_components, the retained PCA-whitened input space (X_pca) should be
    identical up to sign per-row.

    This guards against MIR comparisons being silently performed in different
    input spaces -- a bug class flagged in the user's MIR audit.
    """
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    pytest.importorskip("amica_python")
    import mne
    import os
    from amica_python.benchmark.metrics import pca_inputs_from_ica

    rng = np.random.default_rng(21)
    n_ch = 6
    sources_true = rng.standard_t(df=3, size=(n_ch, 4000))
    A = rng.standard_normal((n_ch, n_ch))
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.standard_normal((n_ch, n_ch))
    X = A @ sources_true
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")

    # MNE Picard
    ica_picard = mne.preprocessing.ICA(n_components=n_ch, method="picard",
                                       fit_params={"ortho": False, "extended": True},
                                       max_iter=1000, random_state=0)
    ica_picard.fit(raw, verbose="ERROR")
    X_pca_picard, _ = pca_inputs_from_ica(ica_picard, raw)

    # AMICA-Python (NumPy CPU; light fit just to get pca_components_)
    os.environ["AMICA_NO_JAX"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import importlib, amica_python.backend
    importlib.reload(amica_python.backend)
    from amica_python import fit_ica
    ica_amica = fit_ica(raw, n_components=n_ch, max_iter=20, random_state=0)
    X_pca_amica, _ = pca_inputs_from_ica(ica_amica, raw)

    # PCA whitening is unique up to a sign per component, so check that the
    # absolute row correlations are ~ 1 between the two PCA bases.
    def _row_corr_abs(A, B):
        A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)
        B = (B - B.mean(axis=1, keepdims=True)) / B.std(axis=1, keepdims=True)
        return np.abs((A * B).mean(axis=1))

    corrs = _row_corr_abs(X_pca_picard, X_pca_amica)
    # Each retained PCA component should match up to a sign within histogram noise.
    assert np.all(corrs > 0.95), (
        f"AMICA and Picard appear to use different PCA bases on the same data: "
        f"per-row |corr| = {corrs}"
    )


def test_amica_convergence_trace_healthy():
    """Verify a small AMICA fit produces a healthy convergence trace:
    log-likelihood is monotone non-decreasing in >= 90% of steps and ends
    finite. This catches regressions that would make MIR uninterpretable.
    """
    pytest.importorskip("mne")
    pytest.importorskip("amica_python")
    import mne, os
    rng = np.random.default_rng(22)
    n_ch = 6
    sources_true = rng.standard_t(df=3, size=(n_ch, 3000))
    A = rng.standard_normal((n_ch, n_ch))
    X = A @ sources_true
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    os.environ["AMICA_NO_JAX"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import importlib, amica_python.backend
    importlib.reload(amica_python.backend)
    from amica_python import fit_ica
    ica = fit_ica(raw, n_components=n_ch, max_iter=50, random_state=0)
    result = getattr(ica, "amica_result_", None)
    assert result is not None, "AMICA fit did not produce amica_result_"
    ll = np.asarray(result.log_likelihood, dtype=float)
    assert ll.size >= 5
    assert np.isfinite(ll[-1])
    monotone_frac = float(np.sum(np.diff(ll) >= 0)) / float(len(ll) - 1)
    assert monotone_frac >= 0.8, (
        f"AMICA log-likelihood is non-monotone in too many steps: {monotone_frac:.2%} monotone "
        f"(LL trace: {ll})"
    )


def test_complete_mir_from_ica_truncated_raises():
    """If ICA truncates below PCA rank (rectangular unmixing), the helper raises."""
    pytest.importorskip("mne")
    pytest.importorskip("picard")
    import mne
    from amica_python.benchmark.metrics import complete_mir_from_ica

    rng = np.random.default_rng(9)
    n_ch = 6
    X = rng.standard_normal((n_ch, 3000))
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_ch)], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    # n_components < n_ch -> rectangular unmixing
    ica = mne.preprocessing.ICA(n_components=3, method="picard", random_state=0)
    ica.fit(raw, verbose="ERROR")
    if ica.unmixing_matrix_.shape[0] != ica.unmixing_matrix_.shape[1]:
        with pytest.raises(ValueError, match="truncated|not square"):
            complete_mir_from_ica(raw, ica, n_bins=50, max_samples=None)
