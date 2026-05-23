"""Ground-truth scoring for the MNE-native synthetic source-recovery benchmark.

Given the true mixing matrix `A_true` (channels x n_true_sources, in
sensor space after the same preprocessing the ICA saw), the true source
waveforms `S_true` (n_true_sources x n_samples, after the same
resampling/filtering as the Raw the ICA saw), and a fitted
`mne.preprocessing.ICA`, compute:

  - Hungarian-matched topography correlations |r_topo_i|
  - Hungarian-matched source-time-course correlations |r_source_i|
  - Normalised Amari index of W_hat @ A_true
  - MIR (mutual information reduction) between input and recovered sources

This module is self-contained: it imports only numpy + scipy and does
not pull in mne (the caller passes plain arrays).

The dispatch / saving logic lives in `run_one_synthetic.py`; the
aggregator (`aggregate_synthetic.py`) just reads the JSON fields this
writes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma


def _zero_mean(arr, axis):
    return arr - arr.mean(axis=axis, keepdims=True)


def _abs_corr_matrix(a, b):
    """|corr| between rows of a (m,k) and rows of b (n,k). Returns (m,n)."""
    a = _zero_mean(np.asarray(a, dtype=np.float64), axis=1)
    b = _zero_mean(np.asarray(b, dtype=np.float64), axis=1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return np.abs((a / a_norm) @ (b / b_norm).T)


def hungarian_match_topographies(A_true, A_hat):
    """Permute & sign-flip columns of A_hat to align with A_true.

    Parameters
    ----------
    A_true : ndarray (n_channels, n_true_sources)
        True mixing matrix in sensor space (average-referenced if the
        downstream pipeline uses average reference).
    A_hat : ndarray (n_channels, n_components)
        Recovered topographies in sensor space (e.g.
        ``ica.get_components()`` from mne.preprocessing.ICA).

    Returns
    -------
    match : dict with keys
        col_idx_in_hat   : (n_true_sources,) index into A_hat columns
        signs            : (n_true_sources,) +/-1 to flip A_hat columns
        r_topo           : (n_true_sources,) signed correlations
        r_topo_abs       : (n_true_sources,) |r| of matched pairs
    """
    A_true = np.asarray(A_true, dtype=np.float64)
    A_hat = np.asarray(A_hat, dtype=np.float64)
    n_true = A_true.shape[1]
    n_hat = A_hat.shape[1]
    if n_hat < n_true:
        raise ValueError(
            f"Need at least n_true ({n_true}) recovered components, "
            f"got {n_hat}.")
    # Build cost = -|corr(A_true_col_i, A_hat_col_j)|; rows = true, cols = recovered.
    cost = -_abs_corr_matrix(A_true.T, A_hat.T)
    row_idx, col_idx = linear_sum_assignment(cost)
    # row_idx will be range(n_true) (square sub-assignment from rectangular cost)
    assert np.array_equal(row_idx, np.arange(n_true))

    # Signed correlation per matched pair -> determines sign-flip
    r_signed = np.zeros(n_true)
    for k in range(n_true):
        a = _zero_mean(A_true[:, k], axis=0)
        b = _zero_mean(A_hat[:, col_idx[k]], axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        r_signed[k] = float((a @ b) / denom) if denom > 0 else 0.0
    signs = np.where(r_signed >= 0, 1, -1).astype(np.int8)
    r_abs = np.abs(r_signed)
    return {
        "col_idx_in_hat": col_idx.astype(int),
        "signs": signs,
        "r_topo": r_signed,
        "r_topo_abs": r_abs,
    }


def score_source_timecourses(S_true, S_hat, col_idx, signs):
    """Per-matched-pair |corr| between true and recovered source signals.

    Parameters
    ----------
    S_true : ndarray (n_true_sources, n_samples)
        True source waveforms (after same resampling/filtering as the
        Raw the ICA was fit on).
    S_hat : ndarray (n_components, n_samples)
        Recovered sources, e.g. ``ica.get_sources(raw).get_data()``.
    col_idx, signs : from ``hungarian_match_topographies``.

    Returns
    -------
    dict
        r_source        : (n_true,) signed correlations
        r_source_abs    : (n_true,) |r|
        rmse_normalised : (n_true,) ||S_true - sign * scale * S_hat[matched]||
                          / ||S_true||  per source
    """
    S_true = np.asarray(S_true, dtype=np.float64)
    S_hat = np.asarray(S_hat, dtype=np.float64)
    n_true = S_true.shape[0]
    if S_hat.shape[1] != S_true.shape[1]:
        raise ValueError(
            f"Sample count mismatch: S_true has {S_true.shape[1]}, "
            f"S_hat has {S_hat.shape[1]}. Run them through the same "
            f"resample/filter chain first.")
    S_true_c = _zero_mean(S_true, axis=1)
    S_hat_picked = S_hat[col_idx] * signs[:, None]
    S_hat_c = _zero_mean(S_hat_picked, axis=1)

    s_true_norm = np.linalg.norm(S_true_c, axis=1)
    s_hat_norm = np.linalg.norm(S_hat_c, axis=1)
    denom = s_true_norm * s_hat_norm
    safe = denom > 0
    r_signed = np.zeros(n_true)
    r_signed[safe] = (S_true_c[safe] * S_hat_c[safe]).sum(axis=1) / denom[safe]

    # Scale-match (LS) then RMSE: scale = <S_true, S_hat> / <S_hat, S_hat>
    scale = np.zeros(n_true)
    s_hat_sq = (S_hat_c * S_hat_c).sum(axis=1)
    scale_safe = s_hat_sq > 0
    scale[scale_safe] = (S_true_c[scale_safe] * S_hat_c[scale_safe]).sum(
        axis=1) / s_hat_sq[scale_safe]
    resid = S_true_c - scale[:, None] * S_hat_c
    rmse_norm = np.zeros(n_true)
    rmse_norm[s_true_norm > 0] = (
        np.linalg.norm(resid[s_true_norm > 0], axis=1) /
        s_true_norm[s_true_norm > 0])

    return {
        "r_source": r_signed,
        "r_source_abs": np.abs(r_signed),
        "rmse_normalised": rmse_norm,
    }


def amari_index(W_hat, A_true):
    """Normalised Amari index of P = W_hat @ A_true.

    Lower is better; 0 = perfect (up to permutation + sign + scale).
    Robust to rectangular ``P`` (n_components may exceed n_true_sources).
    """
    W_hat = np.asarray(W_hat, dtype=np.float64)
    A_true = np.asarray(A_true, dtype=np.float64)
    P = np.abs(W_hat @ A_true)
    n_rows, n_cols = P.shape
    if n_rows < 2 or n_cols < 2:
        return float("nan")
    row_max = P.max(axis=1, keepdims=True)
    col_max = P.max(axis=0, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)
    col_max = np.where(col_max == 0, 1.0, col_max)
    row_term = (P / row_max).sum(axis=1) - 1.0
    col_term = (P / col_max).sum(axis=0) - 1.0
    return float(
        0.5 * (row_term.sum() / (n_cols - 1) +
               col_term.sum() / (n_rows - 1)) /
        max(n_rows, n_cols))


def mir_kbits_per_sec(X_input, X_sources, sfreq, *, max_samples=20_000,
                     k_neighbors=5, random_state=0):
    """Mutual Information Reduction (Kozachenko-Leonenko kNN entropy).

    Matches the definition used by
    ``mne-amica-snapshot/validation/run_validation.py::compute_mir``,
    converted to kbits/s so it lines up with the existing real-data
    benchmark column ``mir_kbits_s``.

    Parameters
    ----------
    X_input : ndarray (n_channels, n_samples)
        Pre-ICA signal (sensor space, same preprocessing as fit).
    X_sources : ndarray (n_components, n_samples)
        Post-ICA sources.
    sfreq : float
        Sampling rate of both signals (Hz).
    max_samples : int
        Cap on samples used for the kNN entropy (memory / runtime).
    k_neighbors : int
        kNN order for the KL estimator.
    random_state : int
        Seed for sample subsampling.

    Returns
    -------
    dict
        bits_per_sample, kbits_per_sec, n_used_components.
    """
    X_input = np.asarray(X_input, dtype=np.float64)
    X_sources = np.asarray(X_sources, dtype=np.float64)
    n_ch = min(X_input.shape[0], X_sources.shape[0])
    n_samples = min(X_input.shape[1], X_sources.shape[1])
    if n_samples > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, max_samples, replace=False)
        idx.sort()
        Xi = X_input[:n_ch, idx]
        Xs = X_sources[:n_ch, idx]
        n_used = max_samples
    else:
        Xi = X_input[:n_ch, :n_samples]
        Xs = X_sources[:n_ch, :n_samples]
        n_used = n_samples

    def entropy_knn(x):
        x = np.sort(x.ravel())
        n = len(x)
        # vectorised k-th nearest-neighbour distance on the sorted line
        # for each i, the kth-nearest neighbour distance is min(
        #     |x[i+k] - x[i]|, |x[i] - x[i-k]| ) when both exist.
        d_fwd = np.full(n, np.inf)
        d_bwd = np.full(n, np.inf)
        if n > k_neighbors:
            d_fwd[:n - k_neighbors] = np.abs(x[k_neighbors:] - x[:n - k_neighbors])
            d_bwd[k_neighbors:] = np.abs(x[k_neighbors:] - x[:n - k_neighbors])
        dists = np.minimum(d_fwd, d_bwd)
        dists = np.maximum(dists, 1e-300)
        # KL estimator in nats
        return digamma(n) - digamma(k_neighbors) + np.log(2) + np.mean(np.log(dists))

    h_input = sum(entropy_knn(Xi[i]) for i in range(n_ch))
    h_sources = sum(entropy_knn(Xs[i]) for i in range(n_ch))
    mir_nats_per_sample = h_input - h_sources
    bits_per_sample = mir_nats_per_sample / np.log(2)
    kbits_per_sec = (bits_per_sample * float(sfreq)) / 1000.0
    return {
        "bits_per_sample": float(bits_per_sample),
        "kbits_per_sec": float(kbits_per_sec),
        "n_used_components": int(n_ch),
        "n_used_samples": int(n_used),
    }


def score_one_fit(A_true, A_hat, S_true, S_hat, W_hat, sfreq):
    """Compute the full ground-truth scoring block for one fit.

    All arrays must already be in the SAME preprocessing state:
      - A_true, A_hat in sensor space, average-referenced if pipeline uses average ref.
      - S_true, S_hat at the same sampling rate (after resampling), same filter, same length.
      - W_hat is the recovered unmixing matrix in the same sensor space as A_true.

    Returns a JSON-serialisable dict.
    """
    topo = hungarian_match_topographies(A_true, A_hat)
    src = score_source_timecourses(
        S_true, S_hat, topo["col_idx_in_hat"], topo["signs"])
    amari = amari_index(W_hat, A_true)
    try:
        mir = mir_kbits_per_sec(A_true @ S_true, S_hat[topo["col_idx_in_hat"]],
                                sfreq)
        mir_block = {
            "mir_vs_truth_bits_per_sample": mir["bits_per_sample"],
            "mir_vs_truth_kbits_s": mir["kbits_per_sec"],
        }
    except Exception as exc:
        mir_block = {"mir_vs_truth_error": str(exc)}

    r_topo = topo["r_topo_abs"]
    r_src = src["r_source_abs"]
    block = {
        "n_true_sources": int(A_true.shape[1]),
        "n_components_recovered": int(A_hat.shape[1]),
        "matched_col_indices": topo["col_idx_in_hat"].tolist(),
        "matched_signs": topo["signs"].tolist(),
        "r_topo": topo["r_topo"].tolist(),
        "r_topo_abs": r_topo.tolist(),
        "r_topo_median": float(np.median(r_topo)),
        "r_topo_min": float(r_topo.min()),
        "r_topo_max": float(r_topo.max()),
        "frac_r_topo_gt_0p95": float((r_topo > 0.95).mean()),
        "frac_r_topo_gt_0p99": float((r_topo > 0.99).mean()),
        "r_source": src["r_source"].tolist(),
        "r_source_abs": r_src.tolist(),
        "r_source_median": float(np.median(r_src)),
        "r_source_min": float(r_src.min()),
        "rmse_source_normalised": src["rmse_normalised"].tolist(),
        "rmse_source_normalised_median": float(np.median(src["rmse_normalised"])),
        "amari_index": amari,
    }
    block.update(mir_block)
    return block
