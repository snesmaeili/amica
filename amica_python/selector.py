"""Surrogate-Calibrated Held-out Likelihood (SCHL) — data-driven AMICA configuration.

Automatic, data-driven selection of AMICA's three manually-set knobs — the PCA rank
``n_components`` (N), the number of models ``M``, and the sample-rejection threshold
``rejsig`` — from the data, with no hand-tuned thresholds and no training corpus.

The whole framework rests on one out-of-sample quantity, :func:`heldout_loglik`: the
mean per-sample marginal log-likelihood of *held-out* data under a *train*-fitted model.
In-sample marginal likelihood is monotone in M and (anti-)monotone under rejection, so it
cannot select either; the held-out likelihood has an interior optimum, and for the
model-order axis its gain is calibrated against a within-data phase-randomized surrogate
null (:func:`phase_randomize`) that preserves the power spectrum and cross-covariance but
destroys temporal non-stationarity.

This module is pure inference + orchestration: it reuses the fitted-model likelihood
engine (``pdf.compute_source_loglikelihood`` / ``multimodel._model_posteriors_from_data``)
and changes no fit/M-step math. Selection *logic* (folds, surrogate null, criteria) is
NumPy/scipy and unit-testable; only :func:`heldout_loglik` touches the (JAX) LL kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- held-out LL
def heldout_loglik(result: Any, X_test: np.ndarray, reduce: str = "mean") -> float:
    """Per-sample marginal log-likelihood of ``X_test`` under a fitted ``result``.

    ``reduce``: ``"mean"`` (default, matches the fit's reported LL — the parity anchor and
    the model-order/rank objective), ``"median"`` (robust to test-block artefacts — the
    rejection objective, so a few contaminated held-out samples don't swamp the signal),
    or ``"none"`` (return the per-sample per-component-normalized array).

    Mirrors the fit's own per-sample LL dispatch (``solver.py`` rejection path): the test
    data is scaled by ``result.data_scale``, mean-removed and whitened with the *train*
    ``whitener_``, then scored under the train model. Single-model uses
    ``compute_source_loglikelihood`` + ``log_det_W`` + ``log_det_sphere``; multi-model uses
    the mixture ``_model_posteriors_from_data`` (``logsumexp_h[log gm_h + log p(x|h)]``).

    Scored entirely in the train's scaled/whitened space, so ``heldout_loglik(result,
    X_train)`` reproduces ``result.log_likelihood[-1]`` to numerical precision when no
    samples were rejected (the parity anchor that validates every convention here).

    Parameters
    ----------
    result : AmicaResult
        A fitted ``solver.AmicaResult`` (single- or multi-model).
    X_test : np.ndarray, shape (n_channels, n_test)
        Held-out data in the SAME channel space the model was fit on.

    Returns
    -------
    float
        Mean held-out per-sample log-likelihood (nats/sample).
    """
    from amica_python.pdf import compute_source_loglikelihood
    from amica_python.likelihood import compute_log_det_W
    from amica_python.multimodel import _model_posteriors_from_data

    S = np.asarray(result.whitener_)                       # (n_comp, n_ch)
    mean = np.asarray(result.mean_).reshape(-1)            # (n_ch,)
    Xs = np.asarray(X_test, dtype=np.float64) * float(result.data_scale)
    data_white = S @ (Xs - mean[:, None])                  # (n_comp, n_test)
    # log|det| of the sphering transform = sum(log(singular values)) = -0.5*sum(log eigs).
    log_det_sphere = float(np.sum(np.log(np.linalg.svd(S, compute_uv=False))))

    W = np.asarray(result.unmixing_matrix_white_)
    alpha = np.asarray(result.alpha_)
    mu = np.asarray(result.mu_)
    beta = np.asarray(result.sbeta_)                       # sbeta_ IS the fit's `beta`
    rho = np.asarray(result.rho_)
    c = np.asarray(result.c_)
    gm = np.atleast_1d(np.asarray(result.gm_))

    if W.ndim == 2:                                        # single model
        c_vec = c.reshape(-1)                              # (n_comp,)
        y = W @ (data_white - c_vec[:, None])
        lls = (np.asarray(compute_source_loglikelihood(y, alpha, mu, beta, rho))
               + float(compute_log_det_W(W)) + log_det_sphere)
    else:                                                  # multi-model mixture
        _P, ll_t, _v, _y = _model_posteriors_from_data(
            data_white, W, c, alpha, mu, beta, rho, gm, log_det_sphere)
        lls = np.asarray(ll_t)
    # Normalize per-component to match the fit's reported log_likelihood convention
    # (nats/sample/component, as in the preprint's DeltaLL); for fixed N this is just a
    # constant factor and does not change argmax over M or rejsig.
    per_sample = np.asarray(lls) / data_white.shape[0]
    if reduce == "mean":
        return float(np.mean(per_sample))
    if reduce == "median":
        return float(np.median(per_sample))
    if reduce == "none":
        return per_sample
    raise ValueError(f"reduce must be 'mean'|'median'|'none', got {reduce!r}")


# ----------------------------------------------------------------- phase-randomized null
def phase_randomize(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multivariate phase-randomized surrogate (Prichard & Theiler, 1994).

    Applies ONE random phase shift per frequency, *shared across all channels*, so every
    channel's power spectrum AND the cross-spectrum (hence the stationary covariance and
    the linear mixing structure ICA sees) are preserved, while temporal non-stationarity
    is destroyed. This is the per-cohort stationary null the model-order criterion
    calibrates against: extra models should not improve held-out fit on it.

    Parameters
    ----------
    X : np.ndarray, shape (n_channels, n_samples)
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
        A real-valued surrogate with the same per-channel spectrum and cross-covariance.
    """
    X = np.asarray(X, dtype=np.float64)
    n_ch, T = X.shape
    Xf = np.fft.rfft(X, axis=1)                            # (n_ch, n_freq)
    n_freq = Xf.shape[1]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    phases[0] = 0.0                                        # DC stays real
    if T % 2 == 0:
        phases[-1] = 0.0                                   # Nyquist stays real (even T)
    rot = np.exp(1j * phases)[None, :]                     # (1, n_freq), shared across ch
    return np.fft.irfft(Xf * rot, n=T, axis=1)


# ------------------------------------------------------------------ cross-validation folds
def block_folds(T: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K contiguous (train_idx, test_idx) folds over a length-T time axis.

    Contiguous blocks (not interleaved) so a held-out fold is a genuine out-of-sample
    *time interval* — the causal, non-stationarity-respecting CV for time series.
    """
    edges = np.linspace(0, T, k + 1).astype(int)
    folds = []
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1])
        train = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], T)])
        folds.append((train, test))
    return folds


def _fit_amica(X, *, num_models=1, rejsig=None, n_components=None, max_iter=300,
               num_mix=3, do_newton=True, seed=0):
    """Thin fit wrapper used by the sweeps (single config -> AmicaResult)."""
    from amica_python import Amica, AmicaConfig
    kw = dict(num_models=int(num_models), max_iter=int(max_iter), num_mix_comps=int(num_mix),
              do_newton=bool(do_newton), do_sphere=True, do_mean=True)
    if n_components is not None:
        kw["pcakeep"] = int(n_components)
    if rejsig is not None:
        kw.update(do_reject=True, rejsig=float(rejsig), rejstart=2, rejint=3, numrej=5)
    return Amica(AmicaConfig(**kw), random_state=int(seed)).fit(X)


# --------------------------------------------------------------------- held-out LL sweeps
def sweep_heldout_ll(X, H_values, folds, *, max_iter=300, num_mix=3, seed=0):
    """Held-out mean LL per model order H, per fold.

    Returns ``{H: [Lho_fold0, Lho_fold1, ...]}`` — fit on each train block, score the
    matching held-out test block with :func:`heldout_loglik`. A fit that fails (e.g. a
    degenerate fold) contributes ``nan`` for that cell.
    """
    Lho = {int(H): [] for H in H_values}
    for (tr, te) in folds:
        Xtr, Xte = X[:, tr], X[:, te]
        for H in H_values:
            try:
                res = _fit_amica(Xtr, num_models=H, max_iter=max_iter, num_mix=num_mix, seed=seed)
                Lho[int(H)].append(heldout_loglik(res, Xte))
            except Exception:
                Lho[int(H)].append(np.nan)
    return Lho


def surrogate_null_increments(X, H_values, folds, *, n_surr=10, max_iter=300,
                              num_mix=3, seed=0, rng=None):
    """Phase-surrogate null distribution of the held-out DeltaLL *increment* per H.

    For each of ``n_surr`` phase-randomized surrogates of the full recording, run the same
    held-out sweep and collect the fold-mean increment ``Lho(H) - Lho(H-1)``. Extra models
    should not help on a (stationary) phase surrogate, so this calibrates "is this H's gain
    real?" without any hand-set threshold. Returns ``{H>=2: [incr_surr0, ...]}``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    Hs = [int(h) for h in H_values]
    incr = {h: [] for h in Hs if h >= 2}
    for _ in range(int(n_surr)):
        Xs = phase_randomize(X, rng)
        Lho_s = sweep_heldout_ll(Xs, Hs, folds, max_iter=max_iter, num_mix=num_mix, seed=seed)
        mean_s = {h: np.nanmean(Lho_s[h]) for h in Hs}
        for h in Hs:
            if h >= 2:
                incr[h].append(mean_s[h] - mean_s[h - 1])
    return incr


def select_model_order(X, *, n_components=None, H_max=6, k_folds=3, n_surr=10, max_iter=300,
                       num_mix=3, kappa_min=25.0, alpha=0.05, seed=0, rng=None):
    """SCHL model-order selection: the largest H whose held-out DeltaLL gain beats the
    phase-surrogate null AND is paired-significant AND respects the kappa data-sufficiency
    bound, with an unbroken admissible prefix chain from H=2 (so it is a stopping rule, not
    an argmax over a noisy plateau).

    Returns ``(M_star, info)`` where ``info`` carries the curves for the SelectionReport.
    """
    from scipy import stats as sstats

    rng = np.random.default_rng(0) if rng is None else rng
    n_ch, T = X.shape
    N = int(n_components or n_ch)
    H_values = list(range(1, int(H_max) + 1))
    folds = block_folds(T, k_folds)
    n_train = int(np.mean([len(tr) for tr, _ in folds]))

    Lho = sweep_heldout_ll(X, H_values, folds, max_iter=max_iter, num_mix=num_mix, seed=seed)
    incr_null = surrogate_null_increments(X, H_values, folds, n_surr=n_surr,
                                          max_iter=max_iter, num_mix=num_mix, seed=seed, rng=rng)
    mean_ho = {h: float(np.nanmean(Lho[h])) for h in H_values}

    rows, M_star = [], 1
    for h in range(2, int(H_max) + 1):
        inc = mean_ho[h] - mean_ho[h - 1]
        null = np.asarray(incr_null.get(h, []), dtype=float)
        null95 = float(np.nanpercentile(null, 95)) if null.size else 0.0
        a = np.asarray(Lho[h], float); b = np.asarray(Lho[h - 1], float)
        ok = np.isfinite(a) & np.isfinite(b)
        per_fold_inc = (a - b)[ok]
        inc = float(np.mean(per_fold_inc)) if per_fold_inc.size else -np.inf
        frac_pos = float(np.mean(per_fold_inc > 0)) if per_fold_inc.size else 0.0
        # Wilcoxon over folds is reported but NOT gated on: with few folds it is
        # underpowered (min p = 0.5 at k=2), so the gate is the surrogate-null calibration
        # plus fold-consistency. The paper's across-SUBJECT comparisons (n~25) use Wilcoxon.
        try:
            p = float(sstats.wilcoxon(a[ok], b[ok], alternative="greater").pvalue) if ok.sum() >= 6 else np.nan
        except Exception:
            p = np.nan
        kappa_ok = n_train >= kappa_min * h * (N ** 2)
        admissible = bool((inc > max(null95, 0.0)) and (frac_pos > 0.5) and kappa_ok)
        rows.append(dict(H=h, increment=inc, null_p95=null95, frac_folds_positive=frac_pos,
                         wilcoxon_p=p, kappa_ok=kappa_ok, admissible=admissible))
        if admissible:
            M_star = h
        else:
            break  # prefix chain: stop at the first non-admissible H
    return M_star, dict(mean_heldout_ll=mean_ho, increments=rows, n_train=n_train, rank=N)


# ------------------------------------------------------------------- rejection threshold
_REJSIG_GRID = (None, 4.0, 3.5, 3.0, 2.75, 2.5, 2.25, 2.0)


def select_rejsig(X, *, num_models=1, n_components=None, rejsig_grid=_REJSIG_GRID,
                  k_folds=3, n_surr=10, max_iter=300, num_mix=3, f_max=0.15, seed=0, rng=None):
    """SCHL rejection selection: the ``rejsig`` (``None`` = off) maximizing held-out
    likelihood, accepted only if its gain over *off* beats a phase-surrogate null and it
    rejects at most ``f_max`` of samples.

    Rejection is a *train-time* estimator choice, so each candidate fits on the train block
    WITH rejection but is scored on the FULL held-out block (no rejection on test). A phase
    surrogate has a thin LL tail, so rejecting it should not help; requiring the real gain
    to exceed that null guards against rejecting genuine (non-artefact) data.

    Returns ``(rejsig_star_or_None, info)``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    T = X.shape[1]
    folds = block_folds(T, k_folds)
    grid = list(rejsig_grid)

    def _profile(data):
        lho, frac = {}, {}
        for rs in grid:
            lls, fr = [], []
            for (tr, te) in folds:
                try:
                    res = _fit_amica(data[:, tr], num_models=num_models, rejsig=rs,
                                     n_components=n_components, max_iter=max_iter,
                                     num_mix=num_mix, seed=seed)
                    lls.append(heldout_loglik(res, data[:, te], reduce="median"))
                    fr.append(0.0 if res.sample_mask_ is None
                              else float(res.n_rejected_) / len(tr))
                except Exception:
                    lls.append(np.nan); fr.append(np.nan)
            lho[rs] = float(np.nanmean(lls)); frac[rs] = float(np.nanmean(fr))
        return lho, frac

    lho, frac = _profile(X)
    off = lho[None]
    cands = [rs for rs in grid if rs is not None and np.isfinite(lho[rs]) and frac[rs] <= f_max]
    info = dict(heldout_ll=lho, rejected_frac=frac, off_ll=off)
    if not cands:
        return None, {**info, "reason": "no candidate within f_max"}
    best = max(cands, key=lambda rs: lho[rs])
    gain = lho[best] - off

    # phase-surrogate null: does the SAME rejsig help on a stationary spectral surrogate?
    null_gains = []
    for _ in range(int(n_surr)):
        Xs = phase_randomize(X, rng)
        gl = []
        for (tr, te) in folds:
            try:
                r0 = _fit_amica(Xs[:, tr], num_models=num_models, rejsig=None,
                                n_components=n_components, max_iter=max_iter, num_mix=num_mix, seed=seed)
                rb = _fit_amica(Xs[:, tr], num_models=num_models, rejsig=best,
                                n_components=n_components, max_iter=max_iter, num_mix=num_mix, seed=seed)
                gl.append(heldout_loglik(rb, Xs[:, te], reduce="median")
                          - heldout_loglik(r0, Xs[:, te], reduce="median"))
            except Exception:
                gl.append(np.nan)
        null_gains.append(float(np.nanmean(gl)))
    null95 = float(np.nanpercentile(null_gains, 95)) if null_gains else 0.0
    info.update(best=best, gain=gain, surrogate_null95=null95, frac_at_best=frac[best])
    if gain > max(null95, 0.0):
        return best, info
    return None, {**info, "reason": "gain did not beat surrogate null"}


# -------------------------------------------------------------------------- PCA rank N
def _full_data_heldout_ll(res, Xtr, Xte, N, *, n_channels):
    """Full-dimensional held-out LL at rank N (per channel), comparable across ranks:
    AMICA on the top-N PCA subspace + a train-fitted Gaussian on the discarded dims.

    The AMICA term is scored in the fit's scaled space (``heldout_loglik`` applies
    ``data_scale``); the residual is scored in the SAME scaled space so the two add."""
    s = float(res.data_scale)
    amica_total = heldout_loglik(res, Xte) * N           # undo per-component norm -> total
    # full PCA of the (scaled, centered) train data; bottom (n_channels-N) dims -> Gaussian
    Xtr_s = Xtr * s
    mu = Xtr_s.mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(Xtr_s))         # ascending
    evals = evals[::-1]; evecs = evecs[:, ::-1]          # descending
    if N >= n_channels:
        return amica_total / n_channels                  # no residual subspace
    Vbot = evecs[:, N:]                                  # (n_ch, n_ch-N)
    var_bot = np.maximum(evals[N:], 1e-12)
    proj = Vbot.T @ (Xte * s - mu)                       # (n_ch-N, T_te)
    gauss = np.mean(np.sum(-0.5 * np.log(2 * np.pi * var_bot)[:, None]
                           - 0.5 * proj ** 2 / var_bot[:, None], axis=0))
    return (amica_total + gauss) / n_channels            # per-channel full-data LL


def select_rank(X, *, N_grid, num_models=1, k_folds=3, max_iter=300, num_mix=3,
                kappa_min=25.0, seed=0):
    """SCHL rank selection: the PCA rank N (from ``N_grid``) maximizing the full-dimensional
    held-out likelihood (AMICA-on-top-N + Gaussian-residual), among ranks that satisfy the
    kappa data-sufficiency bound ``n_train >= kappa_min * num_models * N^2``."""
    n_ch, T = X.shape
    folds = block_folds(T, k_folds)
    n_train = int(np.mean([len(tr) for tr, _ in folds]))
    full_ll = {}
    for N in N_grid:
        vals = []
        for (tr, te) in folds:
            try:
                res = _fit_amica(X[:, tr], num_models=num_models, n_components=int(N),
                                 max_iter=max_iter, num_mix=num_mix, seed=seed)
                vals.append(_full_data_heldout_ll(res, X[:, tr], X[:, te], int(N), n_channels=n_ch))
            except Exception:
                vals.append(np.nan)
        full_ll[int(N)] = float(np.nanmean(vals))
    valid = [int(N) for N in N_grid if n_train >= kappa_min * num_models * (int(N) ** 2)]
    if not valid:
        valid = [int(min(N_grid))]
    N_star = max(valid, key=lambda N: full_ll[N])
    return N_star, dict(full_heldout_ll=full_ll, valid_ranks=valid, n_train=n_train)


# ------------------------------------------------------------------- coordinate ascent
@dataclass
class SelectionReport:
    """The SCHL-selected AMICA configuration + the provenance driving each choice."""
    n_components: int                   # N* (PCA rank)
    num_models: int                     # M*
    rejsig: float | None                # rejsig* (None => rejection off)
    do_reject: bool
    kappa_channels: float               # n_samples / n_channels^2
    kappa_effective: float              # n_samples / N*^2
    rank_info: dict = field(default_factory=dict)
    model_order_info: dict = field(default_factory=dict)
    rejection_info: dict = field(default_factory=dict)

    def fit_params(self) -> dict:
        """The ``fit_params`` dict to hand to ``Amica``/``fit_ica`` for the final fit."""
        p = {"num_models": int(self.num_models)}
        if self.do_reject and self.rejsig is not None:
            p.update(do_reject=True, rejsig=float(self.rejsig))
        return p


def _default_rank_grid(n_ch: int) -> list[int]:
    return sorted({max(2, int(round(f * n_ch))) for f in (0.5, 0.75, 1.0)})


def auto_select_amica(X, *, N_grid=None, H_max=6, rejsig_grid=_REJSIG_GRID, k_folds=5,
                      n_surr=20, max_iter=300, num_mix=3, kappa_min=25.0, seed=0, rng=None):
    """Surrogate-Calibrated Held-out Likelihood (SCHL) configuration of AMICA.

    Coordinate ascent over the three knobs, each by held-out likelihood: (A) PCA rank N at
    M=1 (full-dimensional held-out LL + kappa bound); (B) model order M at N* (held-out
    DeltaLL vs the phase-surrogate null); (C) rejection rejsig at (N*, M*) (robust held-out
    LL + surrogate guard). Returns a :class:`SelectionReport`.

    All inputs are the channel-space data ``X`` (n_channels, n_samples); no manual
    thresholds. Defaults match the cluster grid (k_folds=5, n_surr=20).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_ch, T = X.shape
    grid = list(N_grid) if N_grid is not None else _default_rank_grid(n_ch)

    N_star, rank_info = select_rank(X, N_grid=grid, num_models=1, k_folds=k_folds,
                                    max_iter=max_iter, num_mix=num_mix, kappa_min=kappa_min, seed=seed)
    M_star, mo_info = select_model_order(X, n_components=N_star, H_max=H_max, k_folds=k_folds,
                                         n_surr=n_surr, max_iter=max_iter, num_mix=num_mix,
                                         kappa_min=kappa_min, seed=seed, rng=rng)
    rejsig_star, rej_info = select_rejsig(X, num_models=M_star, n_components=N_star,
                                          rejsig_grid=rejsig_grid, k_folds=k_folds, n_surr=n_surr,
                                          max_iter=max_iter, num_mix=num_mix, seed=seed, rng=rng)
    return SelectionReport(
        n_components=int(N_star), num_models=int(M_star), rejsig=rejsig_star,
        do_reject=rejsig_star is not None,
        kappa_channels=float(T) / float(n_ch ** 2),
        kappa_effective=float(T) / float(N_star ** 2),
        rank_info=rank_info, model_order_info=mo_info, rejection_info=rej_info)
