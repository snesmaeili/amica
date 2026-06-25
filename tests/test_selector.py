"""Tests for the SCHL auto-selection module (amica_python.selector).

Stage 0 (local, CPU): validate the held-out-likelihood spine and the phase-randomized
surrogate on synthetic ground truth before any cluster work.
"""
from __future__ import annotations

import numpy as np
import pytest

from amica_python.selector import heldout_loglik, phase_randomize


# --------------------------------------------------------------------- held-out LL anchor
def test_heldout_loglik_train_parity_m1():
    """heldout_loglik(result, X_train) reproduces the fit's final in-sample mean LL for
    a single-model fit — the anchor that validates the whitening / data_scale / sbeta /
    log_det_sphere conventions all at once (no rejection => full-data LL)."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.default_rng(0)
    n, T = 5, 4000
    A = rng.standard_normal((n, n))
    X = A @ rng.laplace(size=(n, T))  # std ~ O(1) => data_scale stays 1.0

    res = Amica(AmicaConfig(num_models=1, max_iter=150, num_mix_comps=3,
                            do_newton=True, do_sphere=True, do_mean=True),
                random_state=1).fit(X)
    ll_fit = float(res.log_likelihood[-1])
    ll_ho = heldout_loglik(res, X)
    assert np.isfinite(ll_ho)
    assert abs(ll_ho - ll_fit) < 1e-4, f"held-out-on-train {ll_ho:.8f} != fit LL {ll_fit:.8f}"


def test_heldout_loglik_train_parity_m2():
    """Same parity anchor for a multi-model (mixture-LL) fit."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.default_rng(2)
    n, Tseg = 4, 3000
    X = np.concatenate([rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg)),
                        rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg))], axis=1)
    res = Amica(AmicaConfig(num_models=2, max_iter=200, num_mix_comps=3,
                            do_newton=True, do_sphere=True, do_mean=True),
                random_state=1).fit(X)
    ll_fit = float(res.log_likelihood[-1])
    ll_ho = heldout_loglik(res, X)
    assert abs(ll_ho - ll_fit) < 1e-4, f"mm held-out-on-train {ll_ho:.8f} != fit LL {ll_fit:.8f}"


def test_heldout_loglik_generalizes_below_train():
    """On fresh data from the same generative process, held-out LL is finite and not
    wildly above the train LL (a sanity check that the model isn't memorizing)."""
    from amica_python import Amica, AmicaConfig

    rng = np.random.default_rng(7)
    n, T = 5, 4000
    A = rng.standard_normal((n, n))
    X_train = A @ rng.laplace(size=(n, T))
    X_test = A @ rng.laplace(size=(n, T))  # same mixing, fresh sources
    res = Amica(AmicaConfig(num_models=1, max_iter=150, num_mix_comps=3,
                            do_newton=True), random_state=1).fit(X_train)
    ll_train = heldout_loglik(res, X_train)
    ll_test = heldout_loglik(res, X_test)
    assert np.isfinite(ll_test)
    assert ll_test < ll_train + 0.05  # test LL should not exceed train LL by much


# --------------------------------------------------------------- phase-randomized surrogate
def test_phase_randomize_preserves_spectrum_and_covariance():
    """The multivariate phase-randomized surrogate preserves every channel's power
    spectrum (unit-modulus rotation) and the cross-covariance (shared phase => the
    cross-spectrum cancels), to numerical precision."""
    rng = np.random.default_rng(3)
    n, T = 6, 5000
    A = rng.standard_normal((n, n))
    X = A @ rng.standard_normal((n, T))  # correlated channels

    Xs = phase_randomize(X, rng)
    assert Xs.shape == X.shape
    assert np.isrealobj(Xs)
    # per-channel power spectrum preserved
    np.testing.assert_allclose(np.abs(np.fft.rfft(Xs, axis=1)),
                               np.abs(np.fft.rfft(X, axis=1)), rtol=1e-8, atol=1e-6)
    # cross-covariance preserved (Parseval: cov = mean_f X_i conj(X_j), phase cancels)
    np.testing.assert_allclose(np.cov(Xs), np.cov(X), rtol=1e-6, atol=1e-8)


def test_phase_randomize_changes_the_signal():
    """The surrogate is a genuinely different time series (not the identity), so its
    temporal structure is destroyed even though second-order stats are preserved."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 4000))
    Xs = phase_randomize(X, rng)
    assert not np.allclose(Xs, X)


# --------------------------------------------------------------------- model-order selection
def test_select_model_order_detects_two_regimes():
    """Two concatenated distinct ICA mixtures (non-stationary) -> SCHL selects M*>=2:
    the held-out DeltaLL gain at H=2 beats the phase-surrogate null."""
    from amica_python.selector import select_model_order

    rng = np.random.default_rng(20)
    n, Tseg = 4, 2500
    A1, A2 = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    X = np.concatenate([A1 @ rng.laplace(size=(n, Tseg)),
                        A2 @ rng.laplace(size=(n, Tseg))], axis=1)
    # k_folds>=3 so every contiguous train block spans BOTH regimes; with k=2 the two
    # folds align with the two regimes and each train sees only one (CV-on-non-stationary
    # pathology) -- the cluster default is k=5.
    M_star, info = select_model_order(X, H_max=3, k_folds=3, n_surr=3, max_iter=150,
                                      seed=1, rng=np.random.default_rng(0))
    assert M_star >= 2, f"missed non-stationarity: M*={M_star}, rows={info['increments']}"


def test_select_model_order_stationary_returns_one():
    """A single stationary ICA mixture -> SCHL returns M*=1: the held-out gain of a second
    model does not exceed its phase-surrogate null (no over-selection)."""
    from amica_python.selector import select_model_order

    rng = np.random.default_rng(21)
    n, T = 4, 5000
    X = rng.standard_normal((n, n)) @ rng.laplace(size=(n, T))
    M_star, info = select_model_order(X, H_max=3, k_folds=2, n_surr=3, max_iter=150,
                                      seed=1, rng=np.random.default_rng(0))
    assert M_star == 1, f"over-selected on stationary data: M*={M_star}, rows={info['increments']}"


# ------------------------------------------------------------------- rejection selection
def test_select_rejsig_flags_planted_outliers():
    """A clean ICA mixture contaminated with 2% large spikes -> SCHL selects a rejsig
    (not None), and the chosen threshold rejects roughly the planted fraction."""
    from amica_python.selector import select_rejsig

    rng = np.random.default_rng(30)
    n, T = 5, 4000
    X = rng.standard_normal((n, n)) @ rng.laplace(size=(n, T))
    spikes = rng.choice(T, size=T // 50, replace=False)  # 2%
    X[:, spikes] *= 40.0
    rs, info = select_rejsig(X, num_models=1, rejsig_grid=(None, 3.0, 2.5), k_folds=2,
                             n_surr=2, max_iter=80, seed=1, rng=np.random.default_rng(0))
    assert rs is not None, f"rejection not selected on spiky data: {info}"
    assert 0.005 <= info["frac_at_best"] <= 0.12, f"rejected frac out of range: {info['frac_at_best']}"


# --------------------------------------------------------------------------- rank selection
def test_select_rank_recovers_signal_rank():
    """A rank-3 non-Gaussian signal in 6 channels + small noise -> SCHL's full-dimensional
    held-out LL (AMICA-on-top-N + Gaussian-residual) peaks at the true signal rank."""
    from amica_python.selector import select_rank

    rng = np.random.default_rng(31)
    n, r, T = 6, 3, 4000
    A = rng.standard_normal((n, r))
    X = A @ rng.laplace(size=(r, T)) + 0.03 * rng.standard_normal((n, T))
    N_star, info = select_rank(X, N_grid=[2, 3, 4, 5], num_models=1, k_folds=2,
                               max_iter=120, kappa_min=25.0, seed=1)
    assert N_star == r, f"rank N*={N_star} != {r}; curve={info['full_heldout_ll']}"


# ----------------------------------------------------------------- end-to-end coordinate ascent
def test_auto_select_amica_end_to_end():
    """auto_select_amica runs the full coordinate ascent (rank -> model order -> rejection)
    and returns a coherent SelectionReport on non-stationary data (M*>=2)."""
    from amica_python.selector import auto_select_amica

    rng = np.random.default_rng(40)
    n, Tseg = 4, 2500
    X = np.concatenate([rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg)),
                        rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg))], axis=1)
    rep = auto_select_amica(X, N_grid=[3, 4], H_max=2, rejsig_grid=(None, 3.0), k_folds=3,
                            n_surr=2, max_iter=80, seed=1, rng=np.random.default_rng(0))
    assert rep.n_components in (3, 4)
    assert rep.num_models >= 2, f"missed non-stationarity end-to-end: M*={rep.num_models}"
    assert isinstance(rep.do_reject, bool)
    assert rep.fit_params()["num_models"] == rep.num_models
    assert rep.kappa_effective > 0


def test_fit_ica_select_auto():
    """fit_ica(select="auto") runs SCHL on the channel data, fits at the selected config,
    and exposes the SelectionReport on ica.amica_selection_ (MNE end-to-end)."""
    mne = pytest.importorskip("mne")
    from amica_python import fit_ica

    rng = np.random.default_rng(50)
    n, Tseg = 4, 2500
    data = np.concatenate([rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg)),
                           rng.standard_normal((n, n)) @ rng.laplace(size=(n, Tseg))], axis=1) * 1e-6
    info = mne.create_info([f"EEG{i:03d}" for i in range(n)], sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    ica = fit_ica(raw, max_iter=100, select="auto",
                  select_params=dict(N_grid=[3, 4], H_max=2, rejsig_grid=(None, 3.0),
                                     k_folds=3, n_surr=2, max_iter=80, seed=1))
    assert ica.amica_selection_ is not None
    assert ica.amica_selection_.num_models >= 2
    assert ica.n_components_ == ica.amica_selection_.n_components
    assert ica.get_sources(raw).get_data().shape[0] == ica.n_components_

