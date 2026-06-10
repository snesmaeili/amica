"""Information-theoretic metrics for ICA benchmarking (Delorme 2012, Frank 2022/2023/2025).

All entropy / MI quantities here are reported in **bits** (log base 2).
Convert to kbits/sec via `bits_per_sample * sfreq / 1000`.

Design choices, codified for the benchmark:
  * Histogram-based 1D differential entropy estimator (`entropy_histogram`)
    with optional per-row z-scoring + clip. Bias is reported alongside the
    point estimate so callers can sanity-check the n_samples / n_bins ratio.
  * Pairwise MI = sum of marginal entropies - joint entropy, computed from
    a 2D histogram. PMI matrix has 0 on diagonal by definition.
  * Complete MIR follows Frank 2022 eq. (5):
        MIR = sum_i h(x_i) - sum_i h(y_i) + log2|det W|
    where y = W x. When W is rectangular (PCA + ICA), the metric is computed
    in the retained square PCA-whitened subspace and labeled accordingly.
    The function raises if a non-square W is passed without
    `subspace_mode=True`.
  * Remnant PMI = 100 * mean off-diagonal PMI(sources) / mean off-diagonal PMI(input).
    Lower = better. Non-negative by construction (mean PMI is always >= 0).

References:
  - Delorme et al. 2012, "Independent EEG Sources Are Dipolar"
  - Frank, Makeig, Delorme 2022, "A framework to evaluate ICA applied to EEG"
  - Frank et al. 2023, "Optimal parameters for AMICA"
  - Frank et al. 2025, "Quantifying data requirements for EEG ICA using AMICA"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# κ data-sufficiency
# ---------------------------------------------------------------------------

def kappa(n_samples: int, n_channels_or_components: int) -> float:
    """κ = n_samples / n_channels² (Delorme 2012; Frank 2022/2025).

    Target: >= 30 (Delorme 2012 minimum), >= 50 (Frank 2025 paper-grade).
    """
    n = int(n_channels_or_components)
    if n <= 0:
        raise ValueError("n_channels_or_components must be positive")
    return float(n_samples) / float(n * n)


# ---------------------------------------------------------------------------
# Entropy + MI estimators (histogram)
# ---------------------------------------------------------------------------

def _zscore_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (arr - mu) / sd


def entropy_histogram(x: np.ndarray, n_bins: int = 100, clip_sd: float | None = 5.0) -> float:
    """1D differential entropy from a histogram, in bits.

    Returns h(X) = -sum_k p_k * log2(p_k) + log2(bin_width).
    The +log2(bin_width) term reintroduces the continuous reference so
    sum_i h(x_i) - sum_i h(y_i) + log2|det W| is dimensionally correct.

    Parameters
    ----------
    x : 1-D array_like
        Samples of the random variable.
    n_bins : int
        Number of histogram bins (default 100).
    clip_sd : float, optional
        Clip values to +- clip_sd standard deviations before binning. None to
        disable. Default 5.0 (matches Frank 2022's PMI estimator).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return float("nan")
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd <= 0:
        return float("nan")
    if clip_sd is None:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        lo, hi = mu - clip_sd * sd, mu + clip_sd * sd
        x = np.clip(x, lo, hi)
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    p = counts.astype(float) / total
    bin_width = (hi - lo) / n_bins
    if bin_width <= 0:
        return float("nan")
    log2 = np.log(2.0)
    nz = p > 0
    h_disc = -float(np.sum(p[nz] * np.log(p[nz]))) / log2  # in bits
    return h_disc + np.log2(bin_width)


def pairwise_mi_matrix(
    X: np.ndarray,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
    zscore: bool = True,
    max_samples: int | None = 20_000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Pairwise mutual information matrix in **bits**.

    `X[i, t]` is the i-th component / channel time series. Returns an `(n, n)`
    symmetric matrix with zero diagonal. Off-diagonal entries are 2D-histogram
    estimates of `I(x_i; x_j) = h(x_i) + h(x_j) - h(x_i, x_j)`.

    `zscore=True` standardizes each row before binning -- the natural choice
    when comparing channels (volts) to sources (unit-variance), since the
    determinant of the per-row scaling cancels in PMI.
    """
    X = np.asarray(X, dtype=float)
    if zscore:
        X = _zscore_rows(X)
    n, n_t = X.shape
    if max_samples and n_t > max_samples:
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(n_t, max_samples, replace=False))
        X = X[:, idx]
    if clip_sd is not None:
        X = np.clip(X, -clip_sd, clip_sd)
        edges = np.linspace(-clip_sd, clip_sd, n_bins + 1)
    else:
        lo, hi = float(np.nanmin(X)), float(np.nanmax(X))
        edges = np.linspace(lo, hi, n_bins + 1)
    log2 = np.log(2.0)
    pmi = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            hist, _, _ = np.histogram2d(X[i], X[j], bins=(edges, edges))
            total = float(hist.sum())
            if total <= 0:
                continue
            pxy = hist / total
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)
            denom = px * py
            mask = (pxy > 0) & (denom > 0)
            mi_nat = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
            mi_bits = mi_nat / log2
            pmi[i, j] = pmi[j, i] = mi_bits
    return pmi


def mean_pairwise_mi(X: np.ndarray, **kwargs) -> float:
    """Mean off-diagonal PMI in bits. Convenience wrapper."""
    pmi = pairwise_mi_matrix(X, **kwargs)
    n = pmi.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(pmi[mask].mean())


# ---------------------------------------------------------------------------
# Complete MIR (Frank 2022 eq. 7)
# ---------------------------------------------------------------------------

@dataclass
class CompleteMIR:
    """Container for MIR results with provenance."""
    bits_per_sample: float
    kbits_per_sec: float
    h_input_bits: float
    h_sources_bits: float
    log2_abs_det_W: float
    sfreq_hz: float
    n_components: int
    n_samples_used: int
    subspace_mode: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "bits_per_sample": self.bits_per_sample,
            "kbits_per_sec": self.kbits_per_sec,
            "h_input_bits": self.h_input_bits,
            "h_sources_bits": self.h_sources_bits,
            "log2_abs_det_W": self.log2_abs_det_W,
            "sfreq_hz": self.sfreq_hz,
            "n_components": self.n_components,
            "n_samples_used": self.n_samples_used,
            "subspace_mode": self.subspace_mode,
            "note": self.note,
        }


def _validate_mir_inputs(
    X_input: np.ndarray,
    Y_sources: np.ndarray,
    W_square: Optional[np.ndarray],
    subspace_mode: bool,
):
    """Sanity-check inputs to `complete_mir`.

    Raises a clear error if W is not square unless subspace_mode is requested.
    """
    if X_input.ndim != 2 or Y_sources.ndim != 2:
        raise ValueError("X_input and Y_sources must be 2-D (channels/components, time)")
    if X_input.shape[1] != Y_sources.shape[1]:
        raise ValueError(
            "X_input and Y_sources must have the same number of samples "
            f"(got {X_input.shape[1]} vs {Y_sources.shape[1]})"
        )
    if W_square is not None:
        if W_square.ndim != 2:
            raise ValueError("W must be 2-D")
        if W_square.shape[0] != W_square.shape[1] and not subspace_mode:
            raise ValueError(
                f"W has shape {W_square.shape} which is not square; "
                "compute complete MIR in the retained square PCA-whitened subspace "
                "and pass subspace_mode=True. Do NOT compare full-rank and "
                "reduced-rank MIR as if they are identical."
            )


def complete_mir(
    X_input: np.ndarray,
    Y_sources: np.ndarray,
    W_square: Optional[np.ndarray],
    sfreq_hz: float,
    *,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
    max_samples: int | None = 20_000,
    rng_seed: int = 42,
    subspace_mode: bool = False,
    zscore_rows: bool = False,
) -> CompleteMIR:
    """Estimate complete Mutual Information Reduction (MIR) for a linear
    decomposition y = W x, in bits/sample.

    MIR is computed as::

        MIR = log2|det W| + Σ_i h(x_i) − Σ_i h(y_i)

    following the Delorme 2012 / Palmer / Frank 2022 EEG-ICA benchmarking
    definition. The log-determinant term is essential: the entropy-only
    quantity ``Σ h(x_i) − Σ h(y_i)`` is a scale-dependent proxy and must
    not be reported as complete MIR.

    Component gauge / scaling invariance
    ------------------------------------
    If components are rescaled as ``y' = D y`` and W is rescaled
    consistently as ``W' = D W``, true MIR is invariant: the change in
    ``Σ_i h(y_i)`` cancels the change in ``log2|det W|``. Therefore any
    per-row source normalisation must apply the same scaling to both Y
    and W. ``complete_mir_from_ica`` does this automatically.

    Important caveats
    -----------------
    - ``X_input``, ``Y_sources`` and ``W_square`` must refer to the same data
      space. For PCA/rank-reduced ICA, pass the retained PCA-whitened input
      and the retained-rank unmixing matrix, and set ``subspace_mode=True``.
    - For ``W`` rectangular (e.g. when PCA truncates below ICA rank), this
      function will refuse unless ``subspace_mode=True`` AND ``X_input`` /
      ``Y_sources`` are themselves restricted to the retained rank.
    - A negative MIR estimate is **not automatically a scale artifact** once
      gauge invariance is established, but should be treated as a
      diagnostic result. Validate with: identity / permutation / scale
      invariance unit tests, a numerical check that ``Y ≈ W @ X``, healthy
      convergence trace, and a higher-κ run, before drawing quality
      conclusions.

    Set ``zscore_rows=True`` only if you explicitly want a scale-free
    proxy (not true MIR): the determinant term is then zeroed and the
    result is labelled accordingly in ``CompleteMIR.note``.
    """
    X_input = np.asarray(X_input, dtype=float)
    Y_sources = np.asarray(Y_sources, dtype=float)
    _validate_mir_inputs(X_input, Y_sources, W_square, subspace_mode)

    n_components = Y_sources.shape[0]
    n_t = X_input.shape[1]
    if max_samples and n_t > max_samples:
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(n_t, max_samples, replace=False))
        X_sub = X_input[:, idx]
        Y_sub = Y_sources[:, idx]
    else:
        X_sub = X_input
        Y_sub = Y_sources

    if zscore_rows:
        X_sub = _zscore_rows(X_sub)
        Y_sub = _zscore_rows(Y_sub)
        log2_abs_det_W = 0.0
        det_note = "zscore_rows=True -> determinant term zeroed; result is a scale-free PROXY, not full MIR."
    else:
        if W_square is None:
            log2_abs_det_W = 0.0
            det_note = "W not provided; MIR omits the log2|det W| term (acts like entropy_separation)."
        else:
            sign, logabsdet = np.linalg.slogdet(np.asarray(W_square, dtype=float))
            if sign == 0:
                raise ValueError("W is singular (det = 0); cannot compute MIR.")
            log2_abs_det_W = float(logabsdet / np.log(2.0))
            det_note = (
                "subspace MIR (Frank 2022 eq. 7) computed in retained PCA-whitened space."
                if subspace_mode
                else "full-rank MIR (Frank 2022 eq. 7)."
            )

    h_input = float(sum(entropy_histogram(X_sub[i], n_bins=n_bins, clip_sd=clip_sd) for i in range(X_sub.shape[0])))
    h_sources = float(sum(entropy_histogram(Y_sub[i], n_bins=n_bins, clip_sd=clip_sd) for i in range(Y_sub.shape[0])))
    bits_per_sample = h_input - h_sources + log2_abs_det_W
    return CompleteMIR(
        bits_per_sample=float(bits_per_sample),
        kbits_per_sec=float(bits_per_sample * float(sfreq_hz) / 1000.0),
        h_input_bits=float(h_input),
        h_sources_bits=float(h_sources),
        log2_abs_det_W=float(log2_abs_det_W),
        sfreq_hz=float(sfreq_hz),
        n_components=int(n_components),
        n_samples_used=int(X_sub.shape[1]),
        subspace_mode=bool(subspace_mode),
        note=det_note,
    )


def remnant_pmi(
    X_input: np.ndarray,
    Y_sources: np.ndarray,
    *,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
    max_samples: int | None = 20_000,
    rng_seed: int = 42,
) -> dict:
    """remnant_PMI_% = 100 * mean off-diag PMI(sources) / mean off-diag PMI(input).

    Lower is better. Always non-negative (mean PMI >= 0).
    Returns a dict with both means and the ratio for downstream tables.
    """
    pmi_input = mean_pairwise_mi(
        X_input, n_bins=n_bins, clip_sd=clip_sd, zscore=True,
        max_samples=max_samples, rng_seed=rng_seed,
    )
    pmi_source = mean_pairwise_mi(
        Y_sources, n_bins=n_bins, clip_sd=clip_sd, zscore=True,
        max_samples=max_samples, rng_seed=rng_seed,
    )
    if not np.isfinite(pmi_input) or pmi_input <= 0:
        ratio = float("nan")
    else:
        ratio = 100.0 * pmi_source / pmi_input
    return {
        "pmi_input_mean_bits": float(pmi_input),
        "pmi_source_mean_bits": float(pmi_source),
        "remnant_pmi_percent": float(ratio),
        "n_bins": int(n_bins),
        "max_samples": int(max_samples) if max_samples else None,
    }


# ---------------------------------------------------------------------------
# Helpers for hooking the metric functions onto an mne.preprocessing.ICA
# ---------------------------------------------------------------------------

def unmixing_from_ica(ica) -> np.ndarray:
    """Return the unmixing matrix for an ``mne.preprocessing.ICA`` fit.

    For MNE's stored convention this is ``ica.unmixing_matrix_`` directly --
    square in the retained PCA rank space, operating on unwhitened
    ``X_pca = pca_components_ @ centered_data``. Falls back to
    ``pinv(ica.get_components())`` for wrappers that don't expose
    ``unmixing_matrix_``.
    """
    if hasattr(ica, "unmixing_matrix_") and ica.unmixing_matrix_ is not None:
        return np.asarray(ica.unmixing_matrix_, dtype=float)
    mixing = np.asarray(ica.get_components(), dtype=float)
    return np.linalg.pinv(mixing)


def pca_inputs_from_ica(ica, raw):
    """Return (X_pca, W_square) for an mne.preprocessing.ICA in retained rank space.

    MNE stores the **full** PCA basis in ``pca_components_`` (shape n_ch × n_ch)
    but only the top ``n_components_`` rows are passed to ICA. ``unmixing_matrix_``
    is square at ``n_components_ × n_components_`` -- i.e. it operates on the
    retained-rank PCA subspace.

    Returns
    -------
    X_pca : (n_components_, n_t)
        PCA-whitened input restricted to the retained rank.
    W_square : (n_components_, n_components_)
        The ICA unmixing in retained PCA-whitened space (always square here).

    Raises a clear error if `unmixing_matrix_` is not square (would happen if
    a future MNE version truncates ICA below PCA rank).
    """
    if not (hasattr(ica, "pca_components_") and hasattr(ica, "unmixing_matrix_")):
        raise ValueError(
            "ICA object lacks pca_components_ / unmixing_matrix_; cannot compute "
            "complete MIR in retained PCA space. Use the generic complete_mir() "
            "with an explicit W_square instead."
        )
    pca_components = np.asarray(ica.pca_components_, dtype=float)  # (n_ch, n_ch)
    unmixing = np.asarray(ica.unmixing_matrix_, dtype=float)        # (n_comp, n_comp)
    n_comp = unmixing.shape[0]
    if unmixing.shape[0] != unmixing.shape[1]:
        raise ValueError(
            f"unmixing_matrix_ has shape {unmixing.shape} -- ICA truncated below "
            "the retained PCA rank, so complete MIR is not a single square-W operation. "
            "Run with n_components == n_pca, or compute MIR in a smaller retained rank."
        )

    data = np.asarray(raw.get_data(), dtype=float)
    # MNE applies channel-wise pre-whitening (std rescaling) then PCA projection.
    pre_whitener = getattr(ica, "pre_whitener_", None)
    if pre_whitener is not None:
        pw = np.asarray(pre_whitener, dtype=float).reshape(-1, 1)
        data = data / pw
    pca_mean = getattr(ica, "pca_mean_", None)
    if pca_mean is not None:
        data = data - np.asarray(pca_mean, dtype=float).reshape(-1, 1)
    else:
        data = data - data.mean(axis=1, keepdims=True)
    # Take only the top n_comp PCA components (the ones ICA actually saw).
    X_pca = pca_components[:n_comp] @ data  # (n_comp, n_t)
    return X_pca, unmixing


# Back-compat aliases for callers that used the older names.
mne_ica_unmixing_matrix = unmixing_from_ica
_mne_ica_pca_inputs = pca_inputs_from_ica


def complete_mir_from_ica(
    raw,
    ica,
    *,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
    max_samples: int | None = 20_000,
    rng_seed: int = 42,
) -> CompleteMIR:
    """Frank 2022 eq. (5) complete MIR for an mne.preprocessing.ICA fit.

    Computes ::

        MIR = Σ h(X_pca_i) - Σ h(Y_i) + log2|det W|

    in the **retained PCA-whitened rank space**, where
        - X_pca = pca_components_ @ (raw / pre_whitener_ - pca_mean_)
        - Y     = unmixing_matrix_ @ X_pca   (= ica.get_sources(raw))
        - W     = unmixing_matrix_           (square in retained rank space)

    Sources are renormalised to unit variance per row and W is rescaled in
    lockstep. MIR is permutation-scale invariant analytically, so this does
    not change the metric value -- it harmonises the ±5 σ histogram clip
    calibration and the ``log2|det W|`` ledger across methods. MNE wrappers
    around Picard/FastICA/Infomax already produce unit-variance sources;
    AMICA-Python's ``unmixing_matrix_`` does not. Cleaner intermediate values
    (``h_input_bits``, ``h_sources_bits``, ``log2_abs_det_W``) make the JSON
    auditable; a residual MIR gap after this normalisation is a genuine
    decomposition-quality difference, not a gauge artifact.

    Returns a :class:`CompleteMIR` with `.bits_per_sample`, `.kbits_per_sec`,
    `.log2_abs_det_W`, and `subspace_mode=True`. Caller should label any
    downstream report as "MIR in retained rank space" -- do not compare
    full-rank AMICA MIR against this without flagging the subspace.
    """
    X_pca, W_raw = pca_inputs_from_ica(ica, raw)
    Y_raw = W_raw @ X_pca
    # Gauge sources to unit variance per row, propagate the rescaling into W
    # so the |det| term stays consistent. MIR is invariant analytically; this
    # only matters for the histogram clip calibration.
    sigma = Y_raw.std(axis=1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Y = Y_raw / sigma
    W_square = W_raw / sigma  # row-rescale W so Y = W_square @ X_pca holds
    return complete_mir(
        X_pca,
        Y,
        W_square=W_square,
        sfreq_hz=float(raw.info["sfreq"]),
        n_bins=n_bins,
        clip_sd=clip_sd,
        max_samples=max_samples,
        rng_seed=rng_seed,
        subspace_mode=True,
        zscore_rows=False,
    )
