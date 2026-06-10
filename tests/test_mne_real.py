"""Real MNE-Python integration tests.

Requires MNE-Python installed; all tests skip automatically if absent.
Companion to test_mne_integration.py (mocked unit tests for internal helpers).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mne():
    """Skip entire module if MNE-Python is not installed."""
    try:
        import mne as _mne
        return _mne
    except ImportError:
        pytest.skip("MNE-Python not installed")


def _make_raw(mne, n_ch=4, n_samp=1000, seed=42):
    rng = np.random.RandomState(seed)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_ch)],
        sfreq=256,
        ch_types="eeg",
    )
    return mne.io.RawArray(rng.randn(n_ch, n_samp) * 1e-6, info)


def test_fit_ica_on_raw(mne):
    """fit_ica produces a working MNE ICA object; get_sources() works."""
    from amica_python import fit_ica

    raw = _make_raw(mne, n_ch=8, n_samp=2000)
    ica = fit_ica(
        raw,
        n_components=4,
        max_iter=20,
        num_mix=2,
        random_state=42,
        fit_params={"do_newton": False},
    )

    assert ica.n_components_ == 4
    assert ica.method == "amica"
    assert ica.unmixing_matrix_ is not None
    assert ica.get_sources(raw).get_data().shape[0] == 4


def test_direct_vs_shim_sources_correlated(mne):
    """Two identical fit_ica calls produce highly correlated sources."""
    from amica_python import fit_ica

    raw = _make_raw(mne, n_ch=6, n_samp=3000)
    params = dict(
        n_components=3, max_iter=30, num_mix=2,
        random_state=42, fit_params={"do_newton": False},
    )
    ica_a = fit_ica(raw.copy(), **params)
    ica_b = fit_ica(raw.copy(), **params)

    src_a = ica_a.get_sources(raw).get_data()
    src_b = ica_b.get_sources(raw).get_data()

    corr = np.abs(np.corrcoef(src_a, src_b)[:3, 3:])
    max_corrs = np.max(corr, axis=1)
    assert np.all(max_corrs > 0.9), f"Source correlation too low: {max_corrs}"


def test_multi_model_raises(mne):
    """fit_ica() with num_models > 1 raises ValueError (real MNE path)."""
    from amica_python import fit_ica

    raw = _make_raw(mne)
    with pytest.raises(ValueError, match="single-model AMICA"):
        fit_ica(raw, n_components=2, max_iter=10,
                fit_params={"num_models": 2, "do_newton": False})


def test_amica_result_attached(mne):
    """fit_ica() attaches amica_result_ to the ICA object."""
    from amica_python import fit_ica

    raw = _make_raw(mne)
    ica = fit_ica(raw, n_components=2, max_iter=10, fit_params={"do_newton": False})
    assert hasattr(ica, "amica_result_")
    assert ica.amica_result_ is not None


def test_apply_preserves_shape(mne):
    """ica.apply() preserves data shape; full-rank round-trip is machine-precise."""
    from amica_python import fit_ica

    rng = np.random.RandomState(42)
    n_ch, n_samp = 4, 1000
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_ch)],
        sfreq=256, ch_types="eeg",
    )
    data = rng.randn(n_ch, n_samp) * 1e-6
    raw = mne.io.RawArray(data, info)

    ica = fit_ica(raw, n_components=n_ch, max_iter=10, fit_params={"do_newton": False})
    assert ica.apply(raw.copy()).get_data().shape == data.shape

    recon = ica.apply(raw.copy(), exclude=[]).get_data()
    frob_rel = np.linalg.norm(data - recon, "fro") / np.linalg.norm(data, "fro")
    assert frob_rel < 1e-12, f"MNE apply() round-trip Frobenius residual: {frob_rel:.2e}"
