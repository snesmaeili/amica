"""Unit tests for the multi-model stationarity-signature metrics.

Each metric is checked against a case with a known answer (one-hot / uniform
posteriors, a single clean regime switch, perfectly separable trial features).
Pure NumPy + scikit-learn; no JAX, no cluster.
"""
from __future__ import annotations

import numpy as np
import pytest

from amica_python.benchmark import stationarity as st


def test_n_eff_limits():
    assert st.n_eff(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    assert st.n_eff(np.ones(4) / 4) == pytest.approx(4.0, abs=1e-6)
    assert st.n_eff(np.array([0.5, 0.5])) == pytest.approx(2.0, abs=1e-6)


def test_delta_ll():
    out = st.delta_ll({1: -2.0, 2: -1.5, 3: -1.4})
    assert out[1]["delta"] == pytest.approx(0.0)
    assert out[2]["delta"] == pytest.approx(0.5)
    assert out[2]["delta_norm"] == pytest.approx(0.25)  # 0.5 / |−2|
    assert out[3]["delta"] > out[2]["delta"]
    with pytest.raises(ValueError):
        st.delta_ll({2: -1.5})  # missing H=1


def test_posterior_entropy_limits():
    T = 100
    one_hot = np.zeros((3, T)); one_hot[0] = 1.0
    assert np.allclose(st.posterior_entropy_timecourse(one_hot), 0.0, atol=1e-6)
    uniform = np.ones((3, T)) / 3
    assert np.allclose(st.posterior_entropy_timecourse(uniform), np.log(3), atol=1e-6)
    assert st.committed_fraction(one_hot) == pytest.approx(1.0)
    assert st.committed_fraction(uniform) == pytest.approx(0.0)


def test_two_regime_switch_signature():
    """One clean switch at the midpoint → 1 switch, long dwell, near-diagonal P."""
    sfreq = 100.0
    half = 1000
    v = np.zeros((2, 2 * half))
    v[0, :half] = 1.0      # regime 1 → model 0
    v[1, half:] = 1.0      # regime 2 → model 1
    z = st.hard_assignment(v)
    assert z[0] == 0 and z[-1] == 1
    # exactly one switch over 20 s → 0.05 Hz
    assert st.switching_rate(z, sfreq) == pytest.approx(1.0 / 20.0, abs=1e-9)
    # two runs of 10 s each
    assert st.mean_dwell_time(z, sfreq) == pytest.approx(10.0, abs=1e-6)
    P = st.transition_matrix(z, 2)
    assert np.mean(np.diag(P)) > 0.99


def test_stationary_no_switch_signature():
    """A single dominant model everywhere → N_eff≈1, ~zero switching."""
    sfreq = 100.0
    T = 2000
    v = np.zeros((3, T)); v[0] = 0.95; v[1] = 0.03; v[2] = 0.02
    z = st.hard_assignment(v)
    assert st.switching_rate(z, sfreq) == pytest.approx(0.0)
    summ = st.stationarity_summary(np.array([0.95, 0.03, 0.02]), v, sfreq)
    assert summ["n_eff"] < 1.5
    assert summ["transition_diag_mean"] == pytest.approx(1.0)


def test_debounce_removes_jitter():
    sfreq = 100.0
    z = np.zeros(1000, dtype=int)
    z[500] = 1  # single-sample blip (10 ms) — below a 250 ms min dwell
    assert st.switching_rate(z, sfreq, min_dwell_s=0.0) > 0
    assert st.switching_rate(z, sfreq, min_dwell_s=0.25) == pytest.approx(0.0)


def test_mi_argmax_vs_label():
    z = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    assert st.mi_argmax_vs_label(z, y) == pytest.approx(1.0, abs=1e-6)  # perfect
    rng = np.random.default_rng(0)
    zr = rng.integers(0, 2, size=400)
    yr = rng.integers(0, 2, size=400)
    assert st.mi_argmax_vs_label(zr, yr) < 0.1  # independent ⇒ ~0


def test_classify_trial_type_separable_vs_chance():
    """Posteriors that separate labels → high accuracy above chance/permutation."""
    sfreq = 100.0
    rng = np.random.default_rng(1)
    # 30 trials/class, 1 s windows, two regimes cleanly separated in posterior space
    win = 1.0
    n_per = 30
    T = int((2 * n_per + 1) * win * sfreq)
    v = np.full((2, T), 0.5)
    onsets, labels = [], []
    for k in range(2 * n_per):
        on = k * win
        a = int(on * sfreq)
        b = a + int(win * sfreq)
        cls = k % 2
        # class 0 → model 0 dominant; class 1 → model 1 dominant (+ noise)
        p0 = 0.85 if cls == 0 else 0.15
        v[0, a:b] = np.clip(p0 + 0.05 * rng.standard_normal(b - a), 0.01, 0.99)
        v[1, a:b] = 1.0 - v[0, a:b]
        onsets.append(on)
        labels.append("A" if cls == 0 else "B")

    res = st.classify_trial_type(v, sfreq, onsets, labels, window_s=win,
                                 n_splits=5, n_perm=100, random_state=0)
    assert res is not None
    assert res.accuracy > 0.9
    assert res.chance == pytest.approx(0.5, abs=0.05)
    assert res.perm_p < 0.05
    assert res.mi_norm > 0.5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
