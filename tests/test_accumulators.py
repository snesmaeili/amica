"""Direct tests for amica_python.accumulators module."""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_inputs(n_comp=4, n_mix=3, n_chunk=1000):
    rng = np.random.RandomState(42)
    data_chunk = rng.randn(n_comp, n_chunk)
    W = np.eye(n_comp) + 0.01 * rng.randn(n_comp, n_comp)
    alpha = np.ones((n_mix, n_comp)) / n_mix
    mu = rng.randn(n_mix, n_comp) * 0.1
    beta = np.ones((n_mix, n_comp)) + 0.5
    rho = np.full((n_mix, n_comp), 1.5)
    log_det_sphere = 0.5
    return data_chunk, W, alpha, mu, beta, rho, log_det_sphere

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_stats():
    from amica_python.accumulators import zero_stats
    import jax.numpy as jnp

    stats = zero_stats(n_comp=4, n_mix=3)
    
    # Check shape of gy_partial (n_comp, n_comp)
    assert np.asarray(stats.gy_partial).shape == (4, 4)
    assert float(np.sum(np.abs(np.asarray(stats.gy_partial)))) == 0.0

    # Check shape of mu_numer (n_mix, n_comp)
    assert np.asarray(stats.mu_numer).shape == (3, 4)
    assert float(np.sum(np.abs(np.asarray(stats.mu_numer)))) == 0.0

    # Check scalar n_chunk
    assert np.asarray(stats.n_chunk).ndim == 0
    assert float(stats.n_chunk) == 0.0


def test_add_stats():
    from amica_python.accumulators import zero_stats, add_stats
    import jax.numpy as jnp

    stats1 = zero_stats(4, 3)
    stats2 = zero_stats(4, 3)

    # Modify some fields to test addition
    stats1 = stats1._replace(
        n_chunk=jnp.asarray(100.0),
        data_sum=jnp.ones(4) * 2.0
    )
    stats2 = stats2._replace(
        n_chunk=jnp.asarray(50.0),
        data_sum=jnp.ones(4) * 3.0
    )

    stats_sum = add_stats(stats1, stats2)

    assert float(stats_sum.n_chunk) == 150.0
    np.testing.assert_allclose(np.asarray(stats_sum.data_sum), np.ones(4) * 5.0)
    assert float(np.sum(np.abs(np.asarray(stats_sum.gy_partial)))) == 0.0


def test_compute_chunk_stats():
    import jax.numpy as jnp
    from amica_python.accumulators import compute_chunk_stats

    n_comp, n_mix, n_chunk = 4, 3, 500
    (data_chunk, W, alpha, mu, beta, rho, log_det_sphere) = _make_mock_inputs(
        n_comp=n_comp, n_mix=n_mix, n_chunk=n_chunk
    )

    stats = compute_chunk_stats(
        jnp.asarray(data_chunk),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere,
    )

    # Verify scalar properties
    assert float(stats.n_chunk) == float(n_chunk)
    assert np.isfinite(float(stats.ll_sum))

    # Verify shapes
    assert np.asarray(stats.gy_partial).shape == (n_comp, n_comp)
    assert np.asarray(stats.sigma2_partial).shape == (n_comp,)
    assert np.asarray(stats.data_sum).shape == (n_comp,)

    # Mixture shapes
    assert np.asarray(stats.resp_sum).shape == (n_mix, n_comp)
    assert np.asarray(stats.mu_numer).shape == (n_mix, n_comp)
    assert np.asarray(stats.rho_numer).shape == (n_mix, n_comp)

    # All should be finite
    assert np.all(np.isfinite(np.asarray(stats.gy_partial)))
    assert np.all(np.isfinite(np.asarray(stats.mu_denom_le2)))


def test_chunk_stats_additivity():
    """Verify that adding two chunk stats equals computing stats on the full data."""
    import jax.numpy as jnp
    from amica_python.accumulators import compute_chunk_stats, add_stats

    n_comp, n_mix, n_chunk = 4, 3, 1000
    (data, W, alpha, mu, beta, rho, log_det_sphere) = _make_mock_inputs(
        n_comp=n_comp, n_mix=n_mix, n_chunk=n_chunk
    )

    # Compute on full data
    stats_full = compute_chunk_stats(
        jnp.asarray(data), jnp.asarray(W), jnp.asarray(alpha),
        jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho), log_det_sphere
    )

    # Compute on halves
    mid = n_chunk // 2
    stats_h1 = compute_chunk_stats(
        jnp.asarray(data[:, :mid]), jnp.asarray(W), jnp.asarray(alpha),
        jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho), log_det_sphere
    )
    stats_h2 = compute_chunk_stats(
        jnp.asarray(data[:, mid:]), jnp.asarray(W), jnp.asarray(alpha),
        jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho), log_det_sphere
    )

    stats_sum = add_stats(stats_h1, stats_h2)

    # They should match exactly within floating point accuracy
    np.testing.assert_allclose(np.asarray(stats_full.gy_partial), np.asarray(stats_sum.gy_partial), atol=1e-10)
    np.testing.assert_allclose(np.asarray(stats_full.ll_sum), np.asarray(stats_sum.ll_sum), atol=1e-10)
    np.testing.assert_allclose(np.asarray(stats_full.mu_denom_gt2), np.asarray(stats_sum.mu_denom_gt2), atol=1e-10)
    assert float(stats_full.n_chunk) == float(stats_sum.n_chunk)
