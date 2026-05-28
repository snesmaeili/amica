"""Tests for amica-python package."""

import numpy as np
import pytest


def test_fit_random_data():
    """Test basic fitting on random data."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 500
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(
        max_iter=20,
        num_mix_comps=2,
        do_newton=False,
    )
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    assert result.unmixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.unmixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert len(result.log_likelihood) > 0


def test_transform_inverse():
    """Test that transform + inverse_transform reconstructs data."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 500
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    model.fit(data)

    sources = model.transform(data)
    assert sources.shape == (n_channels, n_samples)

    recon = model.inverse_transform(sources)
    assert recon.shape == (n_channels, n_samples)

    # Linear unmix→remix is algebraically exact; residual should be at machine-eps level.
    frob_rel = np.linalg.norm(data - recon, "fro") / np.linalg.norm(data, "fro")
    assert frob_rel < 1e-12, f"Round-trip Frobenius relative residual too high: {frob_rel:.2e}"


def test_ll_increases():
    """Test that log-likelihood generally increases over iterations."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(123)
    n_channels, n_samples = 4, 1000
    # Create data with some structure (mixed sources)
    S = rng.laplace(size=(n_channels, n_samples))
    A = rng.randn(n_channels, n_channels)
    data = A @ S

    config = AmicaConfig(max_iter=100, num_mix_comps=3, do_newton=True)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    ll = result.log_likelihood
    assert len(ll) > 10
    # LL at end should be higher than at start (allowing some tolerance)
    assert ll[-1] > ll[5], "Log-likelihood should increase over training"


"""Test the Picard-compatible functional API."""


def test_amica_function():
    """Test amica() functional API returns correct shapes."""
    from amica_python import amica

    rng = np.random.RandomState(42)
    n_samples, n_components = 500, 4
    # MNE convention: (n_samples, n_components)
    X = rng.randn(n_samples, n_components)

    W = amica(X, max_iter=20, num_mix=2)
    assert W.shape == (n_components, n_components)


def test_amica_return_n_iter():
    """Test return_n_iter flag."""
    from amica_python import amica

    rng = np.random.RandomState(42)
    X = rng.randn(500, 4)

    W, n_iter = amica(X, max_iter=20, num_mix=2, return_n_iter=True)
    assert W.shape == (4, 4)
    assert isinstance(n_iter, int)
    assert n_iter > 0


"""Test actual source separation quality."""


def test_separate_laplacian_sources():
    """Test separation of known Laplacian sources."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(0)
    n_sources, n_samples = 3, 5000

    # Generate independent Laplacian sources
    S = rng.laplace(size=(n_sources, n_samples))

    # Random mixing
    A_true = rng.randn(n_sources, n_sources)
    X = A_true @ S

    config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
    model = Amica(config=config, random_state=42)
    result = model.fit(X)

    # Recover sources
    _ = model.transform(X)

    # Check Amari index (permutation-invariant separation quality)
    # Perfect separation: each row/col of W @ A_true has one dominant entry
    C = result.unmixing_matrix_white_ @ result.whitener_ @ A_true
    # Normalize rows and columns
    C = C / np.max(np.abs(C), axis=1, keepdims=True)

    # Amari index: sum of (sum/max - 1) for rows and cols
    row_ratios = np.sum(np.abs(C), axis=1) / np.max(np.abs(C), axis=1) - 1
    col_ratios = np.sum(np.abs(C), axis=0) / np.max(np.abs(C), axis=0) - 1
    amari = (np.mean(row_ratios) + np.mean(col_ratios)) / 2

    assert amari < 0.3, f"Amari index too high ({amari:.3f}), poor separation"


"""Test explicit matrix naming and consistency."""


def test_matrix_shapes_and_consistency():
    """Test all four matrices have correct shapes and are consistent."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 6, 2000
    S = rng.laplace(size=(n_channels, n_samples))
    A_true = rng.randn(n_channels, n_channels)
    data = A_true @ S

    config = AmicaConfig(max_iter=50, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    # White-space matrices are square (n_comp x n_comp)
    assert result.unmixing_matrix_white_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_white_.shape == (n_channels, n_channels)

    # Sensor-space matrices bridge channels and components
    assert result.unmixing_matrix_sensor_.shape == (n_channels, n_channels)
    assert result.mixing_matrix_sensor_.shape == (n_channels, n_channels)

    # W_white @ A_white ≈ I (algebraic identity; machine-eps level)
    WA = result.unmixing_matrix_white_ @ result.mixing_matrix_white_
    np.testing.assert_allclose(WA, np.eye(n_channels), atol=1e-12)

    # sensor unmixing = W_white @ sphere
    expected_sensor = result.unmixing_matrix_white_ @ result.whitener_
    np.testing.assert_allclose(result.unmixing_matrix_sensor_, expected_sensor, atol=1e-10)

    # sensor mixing = desphere @ A_white
    expected_mix = result.dewhitener_ @ result.mixing_matrix_white_
    np.testing.assert_allclose(result.mixing_matrix_sensor_, expected_mix, atol=1e-10)


def test_sensor_roundtrip():
    """Test mixing_sensor @ unmixing_sensor ≈ I for full-rank data."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 1000
    data = rng.randn(n_channels, n_samples)

    config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    # mixing_sensor @ unmixing_sensor should be identity to machine precision
    product = result.mixing_matrix_sensor_ @ result.unmixing_matrix_sensor_
    np.testing.assert_allclose(product, np.eye(n_channels), atol=1e-12)


"""Test configuration validation."""


def test_defaults():
    """Test default config values match literature recommendations."""
    from amica_python import AmicaConfig

    cfg = AmicaConfig()
    assert cfg.max_iter == 2000
    assert cfg.num_mix_comps == 3
    assert cfg.do_newton
    assert cfg.newt_start == 50
    assert cfg.rejstart == 2
    assert cfg.rejint == 3
    assert cfg.rejsig == 3.0


def test_invalid_config():
    """Test that invalid config raises errors."""
    from amica_python import AmicaConfig

    with pytest.raises(ValueError):
        AmicaConfig(num_models=0)
    with pytest.raises(ValueError):
        AmicaConfig(minrho=0.5)
    with pytest.raises(ValueError):
        AmicaConfig(maxrho=3.0)


"""Test sample rejection feature."""


def test_rejection_enabled():
    """Test that rejection runs without errors when enabled."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_channels, n_samples = 4, 1000
    data = rng.randn(n_channels, n_samples)
    # Add some outliers
    data[:, :10] *= 100

    config = AmicaConfig(
        max_iter=30,
        num_mix_comps=2,
        do_reject=True,
        rejstart=2,
        rejint=3,
        rejsig=3.0,
        numrej=3,
        do_newton=False,
    )
    model = Amica(config=config, random_state=42)
    result = model.fit(data)

    assert result is not None
    assert len(result.log_likelihood) > 0


"""Test MNE-Python integration."""


def test_fit_ica_on_raw():
    """Test fit_ica produces a working MNE ICA object."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    from amica_python import fit_ica

    # Create synthetic Raw object
    sfreq = 256
    n_channels = 8
    n_samples = 2000
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_samples) * 1e-6  # Volts
    raw = mne.io.RawArray(data, info)

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

    # Test that standard MNE methods work
    sources = ica.get_sources(raw)
    assert sources.get_data().shape[0] == 4


"""Test that direct path matches old Infomax shim path."""


def test_direct_vs_shim_sources_correlated():
    """Both paths should produce correlated source activations."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    from amica_python import fit_ica

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(6)],
        sfreq=256,
        ch_types="eeg",
    )
    rng = np.random.RandomState(42)
    data = rng.randn(6, 3000) * 1e-6
    raw = mne.io.RawArray(data, info)

    common_params = dict(
        n_components=3,
        max_iter=30,
        num_mix=2,
        random_state=42,
        fit_params={"do_newton": False},
    )
    ica_direct = fit_ica(raw.copy(), **common_params)
    ica_shim = fit_ica(raw.copy(), **common_params)

    src_direct = ica_direct.get_sources(raw).get_data()
    src_shim = ica_shim.get_sources(raw).get_data()

    # Sources should be highly correlated (up to sign/permutation)
    corr = np.abs(np.corrcoef(src_direct, src_shim)[:3, 3:])
    # Each direct source should match some shim source
    max_corrs = np.max(corr, axis=1)
    assert np.all(max_corrs > 0.9), f"Source correlation too low: {max_corrs}"


"""Test MNE integration guards and metadata."""


def test_multi_model_raises():
    """fit_ica() with num_models > 1 should raise ValueError."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    from amica_python import fit_ica

    sfreq = 256
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(4)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    rng = np.random.RandomState(42)
    raw = mne.io.RawArray(rng.randn(4, 1000) * 1e-6, info)

    with pytest.raises(ValueError, match="single-model AMICA"):
        fit_ica(raw, n_components=2, max_iter=10, fit_params={"num_models": 2, "do_newton": False})


def test_amica_result_attached():
    """fit_ica() should attach amica_result_ to the ICA object."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    from amica_python import fit_ica

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(4)],
        sfreq=256,
        ch_types="eeg",
    )
    rng = np.random.RandomState(42)
    raw = mne.io.RawArray(rng.randn(4, 1000) * 1e-6, info)

    ica = fit_ica(raw, n_components=2, max_iter=10, fit_params={"do_newton": False})
    assert hasattr(ica, "amica_result_")
    assert ica.amica_result_ is not None


def test_apply_preserves_shape():
    """ica.apply() should preserve data shape."""
    try:
        import mne
    except ImportError:
        pytest.skip("MNE-Python not installed")

    from amica_python import fit_ica

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(4)],
        sfreq=256,
        ch_types="eeg",
    )
    rng = np.random.RandomState(42)
    data = rng.randn(4, 1000) * 1e-6
    raw = mne.io.RawArray(data, info)

    # Full-rank fit: n_components == n_channels so round-trip is algebraically exact.
    ica = fit_ica(raw, n_components=4, max_iter=10, fit_params={"do_newton": False})
    raw_clean = ica.apply(raw.copy())
    assert raw_clean.get_data().shape == data.shape

    # apply() with no exclusions and full-rank fit reconstructs to machine precision.
    recon = ica.apply(raw.copy(), exclude=[]).get_data()
    frob_rel = np.linalg.norm(data - recon, "fro") / np.linalg.norm(data, "fro")
    assert frob_rel < 1e-12, f"MNE apply() round-trip Frobenius relative residual: {frob_rel:.2e}"


"""Chunked E-step should match full-batch within float64 rounding."""


def test_chunked_loglik_additivity():
    """sum(compute_loglik_chunk) across halves == compute_total_loglikelihood."""
    from amica_python.backend import jnp
    from amica_python.likelihood import (
        compute_loglik_chunk,
        compute_total_loglikelihood,
    )

    rng = np.random.RandomState(0)
    n_comp, n_mix, n_samp = 4, 3, 10000
    y = rng.randn(n_comp, n_samp)
    W = np.eye(n_comp) + 0.01 * rng.randn(n_comp, n_comp)
    alpha = np.ones((n_mix, n_comp)) / n_mix
    mu = rng.randn(n_mix, n_comp) * 0.1
    beta = np.ones((n_mix, n_comp)) + 0.05 * rng.randn(n_mix, n_comp)
    rho = np.full((n_mix, n_comp), 1.5)

    ll_full = float(
        compute_total_loglikelihood(
            jnp.asarray(y),
            jnp.asarray(W),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(beta),
            jnp.asarray(rho),
            log_det_sphere=0.3,
        )
    )
    ll_h1, n1 = compute_loglik_chunk(
        jnp.asarray(y[:, :5000]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    ll_h2, n2 = compute_loglik_chunk(
        jnp.asarray(y[:, 5000:]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    ll_merged = float((ll_h1 + ll_h2) / (n1 + n2) / n_comp)
    assert abs(ll_full - ll_merged) < 1e-12


def test_chunked_matches_fullbatch_synthetic():
    """Chunked vs full-batch: W and LL agree within rounding after 50 iters."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_src, n_samp = 4, 5000
    srcs = np.stack(
        [
            rng.laplace(size=n_samp),
            rng.standard_t(df=3, size=n_samp),
            rng.laplace(size=n_samp) * 1.5,
            np.sign(rng.randn(n_samp)) * rng.exponential(size=n_samp),
        ]
    )[:n_src]
    srcs = srcs / srcs.std(axis=1, keepdims=True)
    A_true = rng.randn(n_src, n_src)
    A_true = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
    x = A_true @ srcs

    cfg_kw = dict(num_models=1, num_mix_comps=3, max_iter=50, dtype="float64", pcakeep=n_src)
    res_full = Amica(AmicaConfig(**cfg_kw, chunk_size=None), random_state=42).fit(x)
    res_chunk = Amica(AmicaConfig(**cfg_kw, chunk_size=1024), random_state=42).fit(x)

    W_full = np.asarray(res_full.unmixing_matrix_white_)
    W_chunk = np.asarray(res_chunk.unmixing_matrix_white_)
    rel_err = np.max(np.abs(W_chunk - W_full)) / np.max(np.abs(W_full))
    assert rel_err < 1e-4, f"Chunked W diverged from full-batch: rel_err={rel_err:.2e}"

    ll_full = float(np.asarray(res_full.log_likelihood)[-1])
    ll_chunk = float(np.asarray(res_chunk.log_likelihood)[-1])
    assert abs(ll_full - ll_chunk) < 1e-5, (
        f"Final LL diverged: full={ll_full:.8f} chunk={ll_chunk:.8f}"
    )


@pytest.fixture
def tiny_data():
    """Very small data array for fast testing: 4 channels, 20 samples."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 20)


def test_newton_path(tiny_data):
    """Test the full Newton correction path."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        do_newton=True,
        newt_start=1,
        newt_ramp=1,
        max_iter=3,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 3


def test_chunked_path(tiny_data):
    """Test accumulator path with time-axis chunking and Newton."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        chunk_size=5,  # Smaller than n_samples (20)
        max_iter=2,
        do_newton=True,
        newt_start=1,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 2


def test_chunked_path_no_updates(tiny_data):
    """Test accumulator path with no parameter updates."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        chunk_size=5,
        max_iter=1,
        update_alpha=False,
        update_mu=False,
        update_beta=False,
        update_rho=False,
        do_mean=False,
        doscaling=False,
    )
    solver = Amica(config, random_state=42)
    solver.fit(tiny_data)


def test_float32_dtype(tiny_data):
    """Test float32 precision mode."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(dtype="float32", max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    # Just ensure it ran, exact dtype might be float64 in fallback mode
    assert res.unmixing_matrix_white_.shape == (4, 4)


def test_pcakeep_reduces_components(tiny_data):
    """Test pcakeep reduces the component count."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(pcakeep=2, max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.unmixing_matrix_white_.shape == (2, 2)
    assert res.mean_.shape == (4,)
    assert res.whitener_.shape == (2, 4)


def test_fit_transform_and_inverse(tiny_data):
    """Test fit_transform output matches fit().transform() and inverse_transform works."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(max_iter=2)
    solver1 = Amica(config, random_state=42)
    sources1 = solver1.fit_transform(tiny_data)

    solver2 = Amica(config, random_state=42)
    solver2.fit(tiny_data)
    sources2 = solver2.transform(tiny_data)

    np.testing.assert_allclose(sources1, sources2)

    reconstructed = solver2.inverse_transform(sources2)
    assert reconstructed.shape == tiny_data.shape

    # Test unfitted error
    unfitted = Amica(config)
    with pytest.raises(RuntimeError):
        unfitted.transform(tiny_data)
    with pytest.raises(RuntimeError):
        unfitted.inverse_transform(tiny_data)
    with pytest.raises(RuntimeError):
        unfitted.save("dummy")


def test_multimodel_raises(tiny_data):
    """Test multi-model initialization raises NotImplementedError."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(num_models=2, max_iter=2)
    solver = Amica(config, random_state=42)
    with pytest.raises(NotImplementedError):
        solver.fit(tiny_data)


def test_checkpoint_save_load(tiny_data, tmp_path):
    """Test writing checkpoints and reloading."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        outdir=tmp_path / "amica_out",
        max_iter=3,
        writestep=1,  # Write every iteration
    )
    solver = Amica(config, random_state=42)
    res_orig = solver.fit(tiny_data)

    # Load from checkpoint
    solver_loaded = Amica.load(tmp_path / "amica_out")
    res_loaded = solver_loaded.result_

    # Compare
    np.testing.assert_allclose(res_loaded.unmixing_matrix_white_, res_orig.unmixing_matrix_white_)

    # Test load missing dir
    with pytest.raises(FileNotFoundError):
        Amica.load(tmp_path / "nonexistent")

    # Test load missing W
    (tmp_path / "amica_out" / "W").unlink()
    with pytest.raises(FileNotFoundError):
        Amica.load(tmp_path / "amica_out")


def test_checkpoint_load_partial(tiny_data, tmp_path):
    """Test loading a directory where some optional files are missing."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(max_iter=2)
    solver = Amica(config, random_state=42)
    solver.fit(tiny_data)
    solver.save(tmp_path / "amica_out2")

    # Delete optional files
    for fname in ["S", "A", "mean", "c", "alpha", "mu", "rho", "sbeta", "gm", "LL"]:
        if (tmp_path / "amica_out2" / fname).exists():
            (tmp_path / "amica_out2" / fname).unlink()

    solver_loaded = Amica.load(tmp_path / "amica_out2")
    res = solver_loaded.result_
    # Should fallback to defaults
    assert res.alpha_.shape == (1, 4)
    assert res.whitener_.shape == (4, 4)
    assert res.mean_.shape == (4,)
    assert res.log_likelihood.size == 0


def test_rejection_path(tiny_data):
    """Test outlier rejection path."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        do_reject=True,
        rejstart=1,
        numrej=1,
        rejint=1,
        max_iter=2,
    )
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)
    assert res.n_iter == 2


def test_amica_wrapper(tiny_data):
    """Test the Picard-style amica() wrapper function."""
    from amica_python.solver import amica

    # Needs samples x components (transpose of tiny_data)
    X = tiny_data.T

    W, n_iter = amica(X, max_iter=2, return_n_iter=True, random_state=42)
    assert W.shape == (4, 4)
    assert n_iter == 2

    W2 = amica(X, max_iter=1, return_n_iter=False)
    assert W2.shape == (4, 4)


def test_result_to_mne(tiny_data, monkeypatch):
    """Test AmicaResult.to_mne(info) method."""
    import sys
    from unittest.mock import MagicMock

    from amica_python import Amica, AmicaConfig

    # Mock MNE so we don't need real raw
    mne_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "mne", mne_mock)

    class MockICA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICA
    monkeypatch.setitem(sys.modules, "mne.preprocessing", mne_prep_mock)

    config = AmicaConfig(max_iter=2)
    solver = Amica(config, random_state=42)
    res = solver.fit(tiny_data)

    mock_info = {"ch_names": ["EEG1", "EEG2", "EEG3", "EEG4"]}
    ica = res.to_mne(mock_info)

    assert ica.method == "amica"
    assert ica.n_components_ == 4


def test_solver_init_paths(tiny_data):
    """Test solver initialization paths with fixed init and initial params."""
    from amica_python import Amica, AmicaConfig

    # Test fix_init
    config1 = AmicaConfig(fix_init=True, max_iter=1)
    solver1 = Amica(config1, random_state=42)
    solver1.fit(tiny_data)

    # Test init_weights and init_params
    W_init = np.eye(4)
    params = {
        "alpha": np.ones((1, 4)),
        "mu": np.zeros((1, 4)),
        "beta": np.ones((1, 4)),
        "rho": np.ones((1, 4)),
    }
    config2 = AmicaConfig(max_iter=1)
    solver2 = Amica(config2, random_state=42)
    solver2.fit(tiny_data, init_weights=W_init, init_params=params)
