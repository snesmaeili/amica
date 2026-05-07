"""Main AMICA solver class using JAX/NumPy."""

from __future__ import annotations

import logging
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .backend import jax, jnp
from .config import AmicaConfig
from .likelihood import (
    compute_log_det_W,
    compute_total_loglikelihood,
)
from .pdf import (
    compute_all_scores,
    compute_source_loglikelihood,
)
from .preprocessing import (
    preprocess_data,
)
from .updates import (
    apply_full_newton_correction,
    compute_newton_terms,
    update_all_pdf_params,
)

logger = logging.getLogger(__name__)

# Lightweight namedtuple to pass config parameters into the JIT-compiled step.
# Defined at the module level to avoid re-registering a new type on every JIT trace.
ParamConfig = namedtuple(
    "ParamConfig",
    [
        "invsigmin",
        "invsigmax",
        "minrho",
        "maxrho",
        "update_alpha",
        "update_mu",
        "update_beta",
        "update_rho",
        "rholrate",
    ],
)


@partial(
    jax.jit,
    static_argnames=[
        "do_newton",
        "do_mean",
        "do_sphere",
        "doscaling",
        "update_alpha",
        "update_mu",
        "update_beta",
        "update_rho",
    ],
)
def _amica_step(
    # State variables
    W: jnp.ndarray,
    A: jnp.ndarray,
    c: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    gm: jnp.ndarray,
    # Per-iter step size
    lrate_step: float,
    rholrate: float,
    # Static data
    data_white: jnp.ndarray,
    log_det_sphere: float,
    # Config scalars
    newt_start_iter: int,
    iteration: int,
    invsigmin: float,
    invsigmax: float,
    minrho: float,
    maxrho: float,
    # Config flags (static)
    do_newton: bool,
    do_mean: bool,
    do_sphere: bool,
    doscaling: bool,
    update_alpha: bool,
    update_mu: bool,
    update_beta: bool,
    update_rho: bool,
):
    """Single JIT-compiled AMICA iteration step.

    Parameters
    ----------
    W : jnp.ndarray, shape (n_components, n_components)
        Current unmixing matrix.
    A : jnp.ndarray, shape (n_components, n_components)
        Current mixing matrix.
    c : jnp.ndarray, shape (n_components,)
        Current model centers.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Current mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Current mixture locations.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Current mixture inverse scales.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Current mixture shapes.
    gm : jnp.ndarray, shape (n_models,)
        Current model weights (placeholder for future multi-model).
    lrate_step : float
        Learning rate for the natural gradient step.
    rholrate : float
        Learning rate for the shape parameter updates.
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened input data.
    log_det_sphere : float
        Log determinant of the sphering matrix.
    newt_start_iter : int
        Iteration to start applying Newton corrections.
    iteration : int
        Current iteration index.
    invsigmin : float
        Minimum allowed inverse scale parameter.
    invsigmax : float
        Maximum allowed inverse scale parameter.
    minrho : float
        Minimum allowed shape parameter.
    maxrho : float
        Maximum allowed shape parameter.
    do_newton : bool
        Whether to apply Newton correction.
    do_mean : bool
        Whether to center the data via model centers.
    do_sphere : bool
        Whether to use sphered data (affects log-likelihood).
    doscaling : bool
        Whether to re-scale mixing columns to unit norm.
    update_alpha : bool
        Whether to update mixture weights.
    update_mu : bool
        Whether to update mixture locations.
    update_beta : bool
        Whether to update mixture scales.
    update_rho : bool
        Whether to update mixture shapes.

    Returns
    -------
    W_new : jnp.ndarray
        Updated unmixing matrix.
    A_new : jnp.ndarray
        Updated mixing matrix.
    c_new : jnp.ndarray
        Updated model centers.
    alpha_new : jnp.ndarray
        Updated mixture weights.
    mu_new : jnp.ndarray
        Updated mixture locations.
    beta_new : jnp.ndarray
        Updated mixture inverse scales.
    rho_new : jnp.ndarray
        Updated mixture shapes.
    gm_new : jnp.ndarray
        Updated model weights (unchanged for single-model).
    ll : float
        Total log-likelihood of the current state.
    is_good : bool
        Whether the updated parameters are valid (no NaNs).
    newton_used : bool
        Whether the Newton correction was actively applied.
    """

    # 1. E-step: Compute sources
    # y = W * (x - c)
    y = jnp.dot(W, data_white - c[:, None])
    n_samples = y.shape[1]
    n_components = W.shape[0]

    # 2. Compute scores
    g = compute_all_scores(y, alpha, mu, beta, rho)

    # 3. Compute Log-Likelihood
    ll = compute_total_loglikelihood(y, W, alpha, mu, beta, rho, log_det_sphere)

    # 4. Natural Gradient on A
    gy = jnp.dot(g, y.T) / n_samples
    dA_local = jnp.eye(n_components) - gy

    # 5. Newton Correction
    Wtmp = dA_local
    newton_used = jnp.array(False)

    def apply_newton(operands):
        y_, alpha_, mu_, beta_, rho_ = operands
        sigma2, kappa, lambda_ = compute_newton_terms(y_, alpha_, mu_, beta_, rho_)
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
        is_valid = lambda_pos & posdef_newt
        return jnp.where(is_valid, Wtmp_newt, dA_local), is_valid

    if do_newton:

        def try_newton(operands):
            return apply_newton(operands)

        def skip_newton(operands):
            return dA_local, jnp.array(False)

        Wtmp, newton_used = jax.lax.cond(
            iteration >= newt_start_iter, try_newton, skip_newton, (y, alpha, mu, beta, rho)
        )
    else:
        Wtmp = dA_local
        newton_used = jnp.array(False)

    # 6. Update A using step-local lrate (halved natgrad or ramped Newton)
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk

    # 7. Update W = inv(A) — exact inverse via LU.
    def invert_A(A_):
        return jnp.linalg.pinv(A_).astype(W.dtype)

    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jax.lax.cond(
        A_ok,
        invert_A,
        lambda x: W,  # Fallback to old W if A has NaN/Inf
        A_new,
    )
    # Check BOTH A and W for NaN/Inf
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # 8. M-step: Update PDF parameters
    pconfig = ParamConfig(
        invsigmin,
        invsigmax,
        minrho,
        maxrho,
        update_alpha,
        update_mu,
        update_beta,
        update_rho,
        rholrate,
    )

    alpha_new, mu_new, beta_new, rho_new = update_all_pdf_params(
        y, alpha, mu, beta, rho, pconfig, rholrate
    )

    # 9. Update model center.
    if do_mean:
        c_new = jnp.mean(data_white, axis=1)
    else:
        c_new = c

    # 10. Scaling
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        W_new = jnp.linalg.pinv(A_new)

    return (W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, gm, ll, is_good, newton_used)


def _amica_step_chunked(
    W,
    A,
    c,
    alpha,
    mu,
    beta,
    rho,
    gm,
    lrate_step,
    rholrate,
    data_white,
    log_det_sphere,
    newt_start_iter,
    iteration,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton,
    do_mean,
    do_sphere,
    doscaling,
    update_alpha,
    update_mu,
    update_beta,
    update_rho,
    chunk_size: int,
):
    """Chunked-accumulator version of _amica_step for CPU memory scalability.

    Loops over the time axis in chunks of `chunk_size`, accumulates
    sufficient statistics only, then performs a single M-step on the
    totals. Mathematically equivalent to _amica_step up to float64
    rounding (O(eps*T) ≈ 1e-10).

    Not JIT-compiled at the outer level (Python for-loop), but the
    per-chunk `compute_chunk_stats` IS JIT-compiled.

    Parameters
    ----------
    W : jnp.ndarray, shape (n_components, n_components)
        Current unmixing matrix.
    A : jnp.ndarray, shape (n_components, n_components)
        Current mixing matrix.
    c : jnp.ndarray, shape (n_components,)
        Current model centers.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Current mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Current mixture locations.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Current mixture inverse scales.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Current mixture shapes.
    gm : jnp.ndarray, shape (n_models,)
        Current model weights (placeholder for future multi-model).
    lrate_step : float
        Learning rate for the natural gradient step.
    rholrate : float
        Learning rate for the shape parameter updates.
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened input data.
    log_det_sphere : float
        Log determinant of the sphering matrix.
    newt_start_iter : int
        Iteration to start applying Newton corrections.
    iteration : int
        Current iteration index.
    invsigmin : float
        Minimum allowed inverse scale parameter.
    invsigmax : float
        Maximum allowed inverse scale parameter.
    minrho : float
        Minimum allowed shape parameter.
    maxrho : float
        Maximum allowed shape parameter.
    do_newton : bool
        Whether to apply Newton correction.
    do_mean : bool
        Whether to center the data via model centers.
    do_sphere : bool
        Whether to use sphered data (affects log-likelihood).
    doscaling : bool
        Whether to re-scale mixing columns to unit norm.
    update_alpha : bool
        Whether to update mixture weights.
    update_mu : bool
        Whether to update mixture locations.
    update_beta : bool
        Whether to update mixture scales.
    update_rho : bool
        Whether to update mixture shapes.
    chunk_size : int
        Number of samples to process in each chunk.

    Returns
    -------
    W_new : jnp.ndarray
        Updated unmixing matrix.
    A_new : jnp.ndarray
        Updated mixing matrix.
    c_new : jnp.ndarray
        Updated model centers.
    alpha_new : jnp.ndarray
        Updated mixture weights.
    mu_new : jnp.ndarray
        Updated mixture locations.
    beta_new : jnp.ndarray
        Updated mixture inverse scales.
    rho_new : jnp.ndarray
        Updated mixture shapes.
    gm_new : jnp.ndarray
        Updated model weights (unchanged for single-model).
    ll : float
        Total log-likelihood of the current state.
    is_good : bool
        Whether the updated parameters are valid (no NaNs).
    newton_used : bool
        Whether the Newton correction was actively applied.
    """
    from .accumulators import add_stats, compute_chunk_stats, zero_stats
    from .updates import (
        apply_alpha_update_from_stats,
        apply_beta_update_from_stats,
        apply_full_newton_correction,
        apply_mu_update_from_stats,
        apply_rho_update_from_stats,
        compute_newton_terms_from_stats,
    )

    n_samples = data_white.shape[1]
    n_components = W.shape[0]
    n_mix = alpha.shape[0]
    dtype = W.dtype

    # --- E-step: accumulate sufficient statistics over chunks ---
    totals = zero_stats(n_components, n_mix, dtype=dtype)
    data_sum_total = jnp.zeros((n_components,), dtype=dtype)

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        data_chunk = data_white[:, start:stop] - c[:, None]  # pre-center
        stats = compute_chunk_stats(data_chunk, W, alpha, mu, beta, rho, log_det_sphere)
        totals = add_stats(totals, stats)
        # Track uncentered data sum separately for c update
        data_sum_total = data_sum_total + jnp.sum(data_white[:, start:stop], axis=1)

    n_total = float(n_samples)
    ll = totals.ll_sum / n_total / n_components  # match compute_average_loglikelihood

    # --- Natural gradient ---
    gy = totals.gy_partial / n_total
    dA_local = jnp.eye(n_components, dtype=dtype) - gy

    # --- Newton correction ---
    newton_used = jnp.array(False)
    Wtmp = dA_local
    if do_newton and iteration >= newt_start_iter:
        sigma2, kappa, lambda_ = compute_newton_terms_from_stats(
            totals.sigma2_partial,
            totals.resp_sum,
            totals.kappa_numer,
            totals.lambda_numer,
            mu,
            beta,
            n_total,
        )
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
        is_valid = lambda_pos & posdef_newt
        Wtmp = jnp.where(is_valid, Wtmp_newt, dA_local)
        newton_used = is_valid

    # --- A/W update (same as _amica_step) ---
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk
    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jnp.where(A_ok, jnp.linalg.pinv(A_new).astype(dtype), W)
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # --- M-step on PDF params using accumulated stats ---
    if update_alpha:
        alpha_new = apply_alpha_update_from_stats(totals.resp_sum, n_total)
    else:
        alpha_new = alpha

    if update_mu:
        mu_new = apply_mu_update_from_stats(
            mu,
            totals.mu_numer,
            totals.mu_denom_le2,
            totals.mu_denom_gt2,
            rho,
        )
    else:
        mu_new = mu

    if update_beta:
        beta_new = apply_beta_update_from_stats(
            beta,
            totals.resp_sum,
            totals.beta_denom_le2,
            totals.beta_denom_gt2,
            rho,
            invsigmin,
            invsigmax,
        )
    else:
        beta_new = beta

    if update_rho:
        rho_new = apply_rho_update_from_stats(
            rho,
            totals.rho_numer,
            totals.resp_sum,
            rholrate,
            minrho,
            maxrho,
        )
    else:
        rho_new = rho

    # --- c update ---
    if do_mean:
        c_new = data_sum_total / n_total
    else:
        c_new = c

    # --- Column scaling ---
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        W_new = jnp.linalg.pinv(A_new)

    return (
        W_new,
        A_new,
        c_new,
        alpha_new,
        mu_new,
        beta_new,
        rho_new,
        gm,
        ll,
        is_good,
        newton_used,
    )


@dataclass
class AmicaResult:
    """Container for AMICA results.

    Matrix naming convention
    ------------------------
    AMICA operates in whitened space. Matrices are stored in both spaces
    with explicit suffixes to avoid ambiguity:

    - ``*_white_`` — whitened space (after sphering)
    - ``*_sensor_`` — original sensor space

    The relationship is::

        sources = unmixing_matrix_white_ @ whitener_ @ (data - mean_)
        data    = mixing_matrix_sensor_ @ sources + mean_

    Attributes
    ----------
    unmixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Unmixing matrix W in whitened space: ``sources = W @ x_white``.
    mixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Mixing matrix A in whitened space: ``x_white = A @ sources + c``.
    unmixing_matrix_sensor_ : np.ndarray, shape (n_components, n_channels)
        Full unmixing in sensor space: ``W @ sphere``.
    mixing_matrix_sensor_ : np.ndarray, shape (n_channels, n_components)
        Full mixing in sensor space: ``desphere @ A``.
    whitener_ : np.ndarray, shape (n_components, n_channels)
        Sphering/whitening matrix S.
    dewhitener_ : np.ndarray, shape (n_channels, n_components)
        Dewhitening matrix (pseudo-inverse of sphere).
    mean_ : np.ndarray, shape (n_channels,)
        Data mean removed during preprocessing.
    alpha_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Mixture weights for each component.
    mu_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Location parameters.
    rho_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Shape parameters.
    sbeta_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Scale parameters (inverse beta).
    c_ : np.ndarray, shape (n_components,) or (n_models, n_components)
        Model centers.
    gm_ : np.ndarray, shape (n_models,)
        Model weights (for multi-model).
    log_likelihood : np.ndarray, shape (n_iter,)
        Log-likelihood per iteration.
    iteration_times : np.ndarray, shape (n_iter,)
        Wall-clock time per iteration in seconds.
    elapsed_times : np.ndarray, shape (n_iter,)
        Cumulative wall-clock time in seconds.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged.
    """

    unmixing_matrix_white_: np.ndarray
    mixing_matrix_white_: np.ndarray
    unmixing_matrix_sensor_: np.ndarray
    mixing_matrix_sensor_: np.ndarray
    whitener_: np.ndarray
    dewhitener_: np.ndarray
    mean_: np.ndarray
    alpha_: np.ndarray
    mu_: np.ndarray
    rho_: np.ndarray
    sbeta_: np.ndarray
    c_: np.ndarray
    gm_: np.ndarray
    log_likelihood: np.ndarray
    n_iter: int
    iteration_times: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_times: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    data_scale: float = 1.0

    def to_mne(self, info):
        """Convert results to MNE ICA object.

        AMICA decomposes as: ``sources = W @ S @ (x - mean)``
        where W is the unmixing matrix in whitened space and S is the
        sphering/whitening matrix.

        MNE's ICA reconstructs via:
        ``unmixing_full = unmixing_ @ pca_components_[:n_comp]``
        ``mixing_full = pca_components_.T @ mixing_``

        To make these equivalent we use QR decomposition on the combined
        transform ``W @ S`` to extract an orthonormal ``pca_components_``
        (Q.T) and a square ``unmixing_matrix_`` (R), satisfying MNE's
        requirement that ``pca_components_`` has orthonormal rows.

        Parameters
        ----------
        info : mne.Info
            Measurement info (from the Raw/Epochs used for fitting).

        Returns
        -------
        ica : mne.preprocessing.ICA
            Fitted MNE ICA object compatible with plot_components(),
            get_sources(), apply(), and ICLabel.
        """
        try:
            from mne.preprocessing import ICA
        except ImportError as err:
            raise ImportError("MNE-Python is required for to_mne().") from err

        W = np.asarray(self.unmixing_matrix_white_)  # (n_comp, n_comp)
        S = np.asarray(self.whitener_)  # (n_comp, n_ch)
        n_components = W.shape[0]
        n_channels = S.shape[1]

        # Combined transform: sources = WS @ (x - mean)
        WS = W @ S  # (n_comp, n_ch)

        # QR decomposition: WS.T = Q @ R  =>  WS = R.T @ Q.T
        # Q.T is orthonormal (n_comp, n_ch) — use as pca_components_
        # R.T is square (n_comp, n_comp) — use as unmixing_matrix_ (before norms)
        Q, R = np.linalg.qr(WS.T, mode="reduced")  # Q: (n_ch, n_comp), R: (n_comp, n_comp)
        pca_components = Q.T  # (n_comp, n_ch) — orthonormal rows
        unmixing_raw = R.T  # (n_comp, n_comp) — square

        # Verify: WS ≈ unmixing_raw @ pca_components
        # (this is exact by QR construction)

        ica = ICA(n_components=n_components, method="infomax")

        # Build full orthonormal pca_components (n_ch, n_ch).
        # First n_comp rows = Q.T from QR. Complete to orthonormal basis
        # using SVD of Q to get its orthogonal complement.
        U_full, _, Vt_full = np.linalg.svd(Q, full_matrices=True)
        # U_full: (n_ch, n_ch) orthonormal columns
        # First n_comp columns span same space as Q
        # Remaining columns span the null space
        pca_full = U_full.T  # (n_ch, n_ch) — orthonormal rows
        # But we need the first n_comp rows to be exactly Q.T (= pca_components)
        # SVD may reorder/flip signs. Use Q directly and append null space.
        null_space = U_full[:, n_components:]  # (n_ch, n_ch - n_comp)
        pca_full = np.vstack([pca_components, null_space.T])
        ica.pca_components_ = pca_full

        ica.pca_mean_ = np.asarray(self.mean_)
        ica.pre_whitener_ = np.ones((n_channels, 1))

        # pca_explained_variance_ — MNE divides unmixing by sqrt(variance)
        # during fit(). We need to match that convention.
        # Since our pca_components are orthonormal, the "variance" each
        # component explains is the squared column norm of unmixing_raw.
        col_var = np.sum(unmixing_raw**2, axis=0)
        col_var[col_var == 0] = 1.0
        pca_explained_variance = np.ones(n_channels)
        pca_explained_variance[:n_components] = col_var
        ica.pca_explained_variance_ = pca_explained_variance

        # Apply MNE's normalization: unmixing /= sqrt(variance)
        norms = np.sqrt(col_var)
        ica.unmixing_matrix_ = unmixing_raw / norms
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        # Metadata
        ica.n_components_ = n_components
        ica.n_pca_components = n_channels
        ica.info = info
        ica.ch_names = info["ch_names"][:n_channels]
        ica.n_iter_ = self.n_iter
        ica.current_fit = "raw"
        ica.method = "amica"
        ica._is_fitted = True
        ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_components)]

        return ica


class Amica:
    """Native JAX implementation of AMICA algorithm.

    Adaptive Mixture Independent Component Analysis (AMICA) performs ICA
    with adaptive source density modeling using mixtures of generalized
    Gaussians.

    Parameters
    ----------
    config : AmicaConfig, optional
        Configuration object with all algorithm parameters.
        If None, uses default configuration.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    config : AmicaConfig
        Algorithm configuration.
    result_ : AmicaResult
        Fitted model (available after calling fit).

    Examples
    --------
    >>> from amica_python import Amica, AmicaConfig
    >>> config = AmicaConfig(max_iter=500, num_mix_comps=3)
    >>> amica = Amica(config, random_state=42)
    >>> result = amica.fit(data)  # data: (n_channels, n_samples)
    >>> activations = result.unmixing_matrix_white_ @ result.whitener_ @ (data - result.mean_[:, None])

    Notes
    -----
    The AMICA algorithm was developed by Jason Palmer at UCSD.
    This is a native Python/JAX implementation for GPU acceleration.

    References
    ----------
    .. [1] Palmer et al. (2008). Newton method for the ICA mixture model.
           Proc. IEEE ICASSP.
    .. [2] Palmer et al. (2011). AMICA: An adaptive mixture of independent
           component analyzers with shared components. UCSD Technical Report.
    """

    def __init__(
        self,
        config: AmicaConfig | None = None,
        random_state: int | None = None,
    ):
        self.config = config if config is not None else AmicaConfig()
        self.random_state = random_state
        self.rng = jax.random.PRNGKey(random_state if random_state is not None else 0)
        self.result_: AmicaResult | None = None

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"config": self.config, "random_state": self.random_state}

    def set_params(self, **params) -> Amica:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : Amica
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray, shape (n_channels, n_samples)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        X_new : np.ndarray, shape (n_components, n_samples)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def fit(
        self,
        data: np.ndarray,
        init_mean: np.ndarray | None = None,
        init_sphere: np.ndarray | None = None,
        init_weights: np.ndarray | None = None,
        init_params: dict | None = None,
    ) -> AmicaResult:
        """Fit AMICA model to data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Input EEG/MEG data. Should be high-pass filtered.
        init_mean : np.ndarray, optional
            Precomputed mean vector to use instead of computing from data.
        init_sphere : np.ndarray, optional
            Precomputed sphering matrix to use instead of computing from data.
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use for initialization.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.

        Returns
        -------
        result : AmicaResult
            Fitted model containing mixing/unmixing matrices and
            all model parameters.
        """
        if self.config.num_models > 1:
            raise NotImplementedError(
                "Multi-model training is not yet fully implemented in the single-model fit loop."
            )

        # Determine target JAX dtype
        dtype = jnp.float32 if self.config.dtype == "float32" else jnp.float64

        # Preprocessing usually done in float64 for stability, then cast to target dtype
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        # Check for potential unit mismatch (uV vs Volts)
        # Amica (and Engine parity) works best with Volts (std ~ 1e-5 to 1e-4)
        # If std > 0.5, assume uV or similar large unit and auto-scale.
        scaling_factor = 1.0
        data_std = np.std(data)
        if data_std > 1e2:
            # Very large values suggest microvolts or similar units.
            # EEG in Volts: std ~ 1e-5 to 1e-4. In uV: std ~ 10 to 100.
            scaling_factor = 1.0 / data_std
            logger.info(
                "Data std (%.2e) very large. Auto-scaling by %.2e for stability.",
                data_std,
                scaling_factor,
            )
            data = data * scaling_factor

        n_channels, n_samples = data.shape
        logger.info(
            "Fitting %d channels x %d samples using %s", n_channels, n_samples, self.config.dtype
        )

        # Determine number of components
        if self.config.pcakeep is not None:
            n_components = min(self.config.pcakeep, n_channels)
        else:
            n_components = n_channels

        # ========== Preprocessing ==========
        logger.info("Preprocessing (mean removal, sphering)...")
        data_white, mean, sphere, desphere, n_components, eigenvalues = preprocess_data(
            data,
            do_mean=self.config.do_mean,
            do_sphere=self.config.do_sphere,
            pcakeep=self.config.pcakeep,
            mineig=self.config.mineig,
            do_approx=self.config.do_approx_sphere,
            sphere_type=self.config.sphere_type,
            init_mean=init_mean,
            init_sphere=init_sphere,
        )

        # Compute (log, det)(S)| from kept eigenvalues (Fortran sldet)
        safe_eigs = np.maximum(np.asarray(eigenvalues[:n_components]), self.config.mineig)
        self.log_det_sphere = float(-0.5 * np.sum(np.log(safe_eigs)))

        log_det_sphere = self.log_det_sphere

        logger.info("Using %d components", n_components)

        # ========== Initialize Parameters ==========
        W, A, alpha, mu, beta, rho, c, gm = self._initialize_params(
            n_components,
            self.config.num_mix_comps,
            self.config.num_models,
            dtype=dtype,
            init_weights=init_weights,
            init_params=init_params,
        )

        # Store learning rates (can be modified during optimization)
        lrate0 = self.config.lrate
        lrate = lrate0
        newtrate = self.config.newtrate
        rholrate0 = self.config.rholrate
        rholrate = rholrate0
        numdecs = 0
        numincs = 0

        # ========== Main EM Loop ==========
        LL: list[float] = []
        iteration_times: list[float] = []
        elapsed_times: list[float] = []
        converged = False
        newton_count = 0  # Track how many iterations actually used Newton
        natgrad_fallback_count = 0  # Track Newton fallbacks
        start_time = time.perf_counter()

        # Initial ll_prev for first iteration
        ll_prev_val = -np.inf

        # Sample rejection state
        rej_count = 0  # Number of rejection passes done

        # Convert config flags to static arguments once
        do_newton_static = self.config.do_newton
        do_mean_static = self.config.do_mean
        do_sphere_static = self.config.do_sphere
        doscaling_static = self.config.doscaling
        update_alpha_static = self.config.update_alpha
        update_mu_static = self.config.update_mu
        update_beta_static = self.config.update_beta
        update_rho_static = self.config.update_rho

        # Ensure initial state is JAX array with correct dtype
        W = jnp.asarray(W, dtype=dtype)
        A = jnp.asarray(A, dtype=dtype)
        c = jnp.asarray(c, dtype=dtype)
        alpha = jnp.asarray(alpha, dtype=dtype)
        mu = jnp.asarray(mu, dtype=dtype)
        beta = jnp.asarray(beta, dtype=dtype)
        rho = jnp.asarray(rho, dtype=dtype)
        gm = jnp.asarray(gm, dtype=dtype)
        data_white = jnp.asarray(data_white, dtype=dtype)

        for iteration in range(self.config.max_iter):
            iter_start = time.perf_counter()

            # Stage 1 — decay on LL decrease
            if iteration > 0 and len(LL) >= 2 and LL[-1] < LL[-2]:
                if lrate <= self.config.minlrate:
                    logger.info("Converged at iteration %d (lrate <= minlrate)", iteration)
                    converged = True
                    break
                lrate = lrate * self.config.lratefact
                rholrate = rholrate * self.config.rholratefact
                numdecs += 1
                if numdecs >= self.config.max_decs:
                    lrate0 = lrate0 * self.config.lratefact
                    if iteration > self.config.newt_start:
                        rholrate0 = rholrate0 * self.config.rholratefact
                    if self.config.do_newton and iteration > self.config.newt_start:
                        newtrate = newtrate * self.config.lratefact
                    numdecs = 0
            elif iteration > 0 and len(LL) >= 2 and LL[-1] > LL[-2]:
                numdecs = 0

            # Stage 2 — per-iter ramp toward ceiling
            in_newton = self.config.do_newton and (iteration >= self.config.newt_start)
            ceiling = newtrate if in_newton else lrate0
            lrate = min(ceiling, lrate + min(1.0 / self.config.newt_ramp, lrate))

            # Dispatch: full-batch (default) or chunked E-step
            if self.config.chunk_size is not None:
                cs = self.config.chunk_size

                def _step_fn(*args, cs=cs):
                    return _amica_step_chunked(*args, chunk_size=cs)

            else:
                _step_fn = _amica_step
            (
                W_new,
                A_new,
                c_new,
                alpha_new,
                mu_new,
                beta_new,
                rho_new,
                gm_new,
                ll_curr,
                is_good,
                newton_used,
            ) = _step_fn(
                W,
                A,
                c,
                alpha,
                mu,
                beta,
                rho,
                gm,
                lrate,
                rholrate,
                data_white,
                log_det_sphere,
                # Config scalars
                self.config.newt_start,
                iteration,
                self.config.invsigmin,
                self.config.invsigmax,
                self.config.minrho,
                self.config.maxrho,
                # Config flags (static)
                do_newton_static,
                do_mean_static,
                do_sphere_static,
                doscaling_static,
                update_alpha_static,
                update_mu_static,
                update_beta_static,
                update_rho_static,
            )

            # Block until scalars are ready (synchronize for checking)
            is_good_val = bool(is_good)
            ll_val = float(ll_curr)
            newton_used_val = bool(newton_used)

            if not is_good_val:
                lrate *= 0.5
                if iteration % 10 == 0:
                    logger.warning(
                        "Iter %d: NaN/Inf detected, reducing lrate to %.2e", iteration, lrate
                    )
                iteration_times.append(time.perf_counter() - iter_start)
                elapsed_times.append(time.perf_counter() - start_time)
                continue

            # Accept update
            W, A, c, alpha, mu, beta, rho, gm = (
                W_new,
                A_new,
                c_new,
                alpha_new,
                mu_new,
                beta_new,
                rho_new,
                gm_new,
            )
            LL.append(ll_val)

            # ========== Sample Rejection ==========
            if (
                self.config.do_reject
                and iteration >= self.config.rejstart
                and rej_count < self.config.numrej
                and (iteration - self.config.rejstart) % self.config.rejint == 0
            ):
                # Compute per-sample LL
                y_rej = jnp.dot(W, data_white - c[:, None])
                sample_lls = compute_source_loglikelihood(y_rej, alpha, mu, beta, rho)
                sample_lls = sample_lls + compute_log_det_W(W) + log_det_sphere

                # Rejection threshold: mean - rejsig * std
                ll_mean = jnp.mean(sample_lls)
                ll_std = jnp.std(sample_lls)
                threshold = ll_mean - self.config.rejsig * ll_std

                new_mask = sample_lls >= threshold
                n_rejected = int(jnp.sum(~new_mask))
                max_reject = int(0.2 * n_samples)  # Cap at 20%

                if n_rejected <= max_reject:
                    rej_count += 1
                    if iteration % 10 == 0:
                        pct = 100.0 * n_rejected / n_samples
                        logger.info(
                            "Iter %d: Rejected %d samples (%.1f%%)", iteration, n_rejected, pct
                        )

                    # Subset data to non-rejected samples
                    mask_np = np.asarray(new_mask)
                    kept_idx = np.where(mask_np)[0]
                    data_white = jnp.asarray(np.asarray(data_white)[:, kept_idx])
                    n_samples = data_white.shape[1]

            dll = ll_val - ll_prev_val
            if iteration > 0 and self.config.use_min_dll:
                if dll < self.config.min_dll:
                    numincs += 1
                    if numincs > self.config.max_incs:
                        logger.info("Converged at iteration %d (dll < min_dll)", iteration)
                        converged = True
                        iteration_times.append(time.perf_counter() - iter_start)
                        elapsed_times.append(time.perf_counter() - start_time)
                        break
                else:
                    numincs = 0

            # Track Newton usage
            if iteration >= self.config.newt_start and do_newton_static:
                if newton_used_val:
                    newton_count += 1
                else:
                    natgrad_fallback_count += 1

            # Progress output
            if iteration % 10 == 0:
                if iteration >= self.config.newt_start and do_newton_static:
                    mode = "N" if newton_used_val else "ng"
                    logger.info(
                        "Iter %4d: LL = %.6f, lrate = %.2e [%s]", iteration, ll_val, lrate, mode
                    )
                else:
                    logger.info("Iter %4d: LL = %.6f, lrate = %.2e", iteration, ll_val, lrate)

            # Checkpoint
            if (
                self.config.outdir is not None
                and self.config.writestep > 0
                and (iteration + 1) % self.config.writestep == 0
            ):
                # Need to bring back to CPU for saving
                self.result_ = AmicaResult(
                    unmixing_matrix_white_=np.asarray(W),
                    mixing_matrix_white_=np.asarray(A),
                    unmixing_matrix_sensor_=np.asarray(W @ sphere),
                    mixing_matrix_sensor_=np.asarray(desphere @ A),
                    whitener_=np.asarray(sphere),
                    dewhitener_=np.asarray(desphere),
                    mean_=np.asarray(mean),
                    alpha_=np.asarray(alpha),
                    mu_=np.asarray(mu),
                    rho_=np.asarray(rho),
                    sbeta_=np.asarray(beta),
                    c_=np.asarray(c),
                    gm_=np.asarray(gm),
                    log_likelihood=np.array(LL),
                    iteration_times=np.array(iteration_times),
                    elapsed_times=np.array(elapsed_times),
                    n_iter=len(LL),
                    converged=converged,
                    data_scale=scaling_factor,
                )
                self.save(self.config.outdir)
                logger.info("Saved checkpoint to %s", self.config.outdir)

            ll_prev_val = ll_val
            iteration_times.append(time.perf_counter() - iter_start)
            elapsed_times.append(time.perf_counter() - start_time)

        if not converged:
            logger.info("Reached max_iter (%d)", self.config.max_iter)

        # Newton diagnostic summary
        if do_newton_static and self.config.newt_start < len(LL):
            total_newton_iters = newton_count + natgrad_fallback_count
            if total_newton_iters > 0:
                pct = 100.0 * newton_count / total_newton_iters
                logger.info(
                    "Newton: %d/%d iterations used Newton (%.0f%%), "
                    "%d fell back to natural gradient",
                    newton_count,
                    total_newton_iters,
                    pct,
                    natgrad_fallback_count,
                )

        # ========== Construct Result ==========
        self.result_ = AmicaResult(
            unmixing_matrix_white_=np.asarray(W),
            mixing_matrix_white_=np.asarray(A),
            unmixing_matrix_sensor_=np.asarray(W @ sphere),
            mixing_matrix_sensor_=np.asarray(desphere @ A),
            whitener_=np.asarray(sphere),
            dewhitener_=np.asarray(desphere),
            mean_=np.asarray(mean),
            alpha_=np.asarray(alpha),
            mu_=np.asarray(mu),
            rho_=np.asarray(rho),
            sbeta_=np.asarray(beta),
            c_=np.asarray(c),
            gm_=np.asarray(gm),
            log_likelihood=np.array(LL),
            iteration_times=np.array(iteration_times),
            elapsed_times=np.array(elapsed_times),
            n_iter=len(LL),
            converged=converged,
            data_scale=scaling_factor,
        )

        assert self.result_ is not None
        return self.result_

    def _initialize_params(
        self,
        n_components: int,
        n_mix: int,
        n_models: int,
        dtype: Any = jnp.float64,
        init_weights: np.ndarray | None = None,
        init_params: dict | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        """Initialize model parameters.

        Parameters
        ----------
        n_components : int
            Number of ICA components.
        n_mix : int
            Number of Gaussian mixture components.
        n_models : int
            Number of ICA models.
        dtype : Any
            JAX dtype for parameters (jnp.float32 or jnp.float64).
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.

        Returns
        -------
        W : jnp.ndarray, shape (n_components, n_components)
            Initial unmixing matrix.
        A : jnp.ndarray, shape (n_components, n_components)
            Initial mixing matrix (inverse of W).
        alpha : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture weights.
        mu : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture locations.
        beta : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture inverse scales (sbeta).
        rho : jnp.ndarray, shape (n_mix, n_components)
            Initial mixture shape parameters.
        c : jnp.ndarray, shape (n_components,)
            Initial model centers.
        gm : jnp.ndarray, shape (n_models,)
            Initial model probabilities.
        """
        rng = np.random.default_rng(self.random_state)

        # Initialize mixing matrix then invert to get W.
        if init_weights is not None:
            W = jnp.asarray(init_weights, dtype=dtype)
            A = jnp.linalg.pinv(W)
        else:
            if self.config.fix_init:
                A_np = np.eye(n_components, dtype=np.float64)
            else:
                noise = rng.random((n_components, n_components))
                A_np = 0.01 * (0.5 - noise)
                A_np += np.eye(n_components, dtype=np.float64)
                col_norms = np.linalg.norm(A_np, axis=0)
                col_norms = np.where(col_norms > 0.0, col_norms, 1.0)
                A_np = A_np / col_norms
            A = jnp.asarray(A_np, dtype=dtype)
            W = jnp.linalg.pinv(A)

        # Initialize mixture parameters
        if init_params is not None and "alpha" in init_params:
            alpha = jnp.asarray(init_params["alpha"], dtype=dtype)
        else:
            # alpha: uniform mixture weights
            alpha = jnp.ones((n_mix, n_components), dtype=dtype) / n_mix

        if init_params is not None and "mu" in init_params:
            mu = jnp.asarray(init_params["mu"], dtype=dtype)
        else:
            base = np.arange(n_mix, dtype=np.float64) - (n_mix - 1) / 2.0
            mu_np = base[:, None] * np.ones((n_mix, n_components), dtype=np.float64)
            if not self.config.fix_init:
                noise = rng.random((n_mix, n_components))
                mu_np = mu_np + 0.05 * (1.0 - 2.0 * noise)
            mu = jnp.asarray(mu_np, dtype=dtype)

        if init_params is not None and ("beta" in init_params or "sbeta" in init_params):
            beta_key = "beta" if "beta" in init_params else "sbeta"
            beta = jnp.asarray(init_params[beta_key], dtype=dtype)
        else:
            if self.config.fix_init:
                beta = jnp.ones((n_mix, n_components), dtype=dtype)
            else:
                noise = rng.random((n_mix, n_components))
                beta = jnp.asarray(1.0 + 0.1 * (0.5 - noise), dtype=dtype)

        if init_params is not None and "rho" in init_params:
            rho = jnp.asarray(init_params["rho"], dtype=dtype)
        else:
            # rho: start at middle value
            rho = jnp.full((n_mix, n_components), self.config.rho0, dtype=dtype)

        # Model center: zero
        c = jnp.zeros(n_components, dtype=dtype)

        # Model weights: uniform
        gm = jnp.ones(n_models, dtype=dtype) / n_models

        return W, A, alpha, mu, beta, rho, c, gm

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted unmixing to new data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            New data to transform.

        Returns
        -------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data = np.asarray(data, dtype=np.float64) * self.result_.data_scale

        # Apply full unmixing: W @ sphere @ (x - mean)
        centered = data - self.result_.mean_[:, None]
        whitened = self.result_.whitener_ @ centered
        sources = self.result_.unmixing_matrix_white_ @ whitened

        return sources

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct data from sources.

        Parameters
        ----------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.

        Returns
        -------
        data : np.ndarray, shape (n_channels, n_samples)
            Reconstructed data.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        sources = np.asarray(sources, dtype=np.float64)

        # data = A @ sources + mean = desphere @ A_white @ sources + mean
        data = self.result_.mixing_matrix_sensor_ @ sources + self.result_.mean_[:, None]

        return data / self.result_.data_scale

    def save(self, outdir: str | Path) -> None:
        """Save model to directory in AMICA-compatible format.

        Parameters
        ----------
        outdir : str or Path
            Output directory.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save in Fortran-compatible binary format (column-major)
        def save_binary(name: str, arr: np.ndarray):
            arr.astype("<f8").T.tofile(outdir / name)

        save_binary("A", self.result_.mixing_matrix_sensor_)
        save_binary("W", self.result_.unmixing_matrix_white_)
        save_binary("S", self.result_.whitener_)
        save_binary("mean", self.result_.mean_)
        save_binary("alpha", self.result_.alpha_)
        save_binary("mu", self.result_.mu_)
        save_binary("rho", self.result_.rho_)
        save_binary("sbeta", self.result_.sbeta_)
        save_binary("c", self.result_.c_)
        save_binary("gm", self.result_.gm_)
        save_binary("LL", self.result_.log_likelihood)

        logger.info("Saved model to %s", outdir)

    @classmethod
    def load(cls, outdir: str | Path) -> Amica:
        """Load model from AMICA-compatible directory.

        Parameters
        ----------
        outdir : str or Path
            Input directory containing AMICA binary files.

        Returns
        -------
        model : Amica
            Loaded AMICA model.
        """
        outdir = Path(outdir)
        if not outdir.exists():
            raise FileNotFoundError(f"Directory {outdir} does not exist")

        # Helper to read binary double files
        def read_bin(name, shape=None):
            p = outdir / name
            if not p.exists():
                return None
            # AMICA writes doubles (float64) in column-major order
            data = np.fromfile(p, dtype=np.float64)
            if shape is not None:
                data = data.reshape(shape, order="F")
            return data

        # Load parameters
        W = read_bin("W")
        if W is None:
            raise FileNotFoundError(f"Could not find W in {outdir}")

        # W is n_components x n_components
        n_components = int(np.sqrt(W.size))
        W = W.reshape((n_components, n_components), order="F")

        S = read_bin("S", (n_components, n_components))
        if S is None:
            S = np.eye(n_components)

        A = read_bin("A", (n_components, n_components))
        if A is None:
            A = np.linalg.pinv(W)

        mean = read_bin("mean", (n_components,))
        if mean is None:
            mean = np.zeros(n_components)

        c = read_bin("c", (n_components,))
        if c is None:
            c = np.zeros(n_components)

        # Check size of alpha to infer n_mix
        alpha_raw = read_bin("alpha")
        if alpha_raw is not None:
            n_mix = alpha_raw.size // n_components
            alpha = alpha_raw.reshape((n_mix, n_components), order="F")
            mu = read_bin("mu", (n_mix, n_components))
            sbeta = read_bin("sbeta", (n_mix, n_components))
            rho = read_bin("rho", (n_mix, n_components))
        else:
            n_mix = 1
            alpha = np.ones((n_mix, n_components))
            mu = np.zeros((n_mix, n_components))
            sbeta = np.ones((n_mix, n_components))
            rho = np.ones((n_mix, n_components)) * 1.5

        gm = read_bin("gm", (1,))
        if gm is None:
            gm = np.ones(1)

        LL = read_bin("LL")
        if LL is None:
            LL = np.array([])

        result = AmicaResult(
            unmixing_matrix_white_=W,
            mixing_matrix_white_=np.linalg.pinv(W),
            unmixing_matrix_sensor_=W @ S,
            mixing_matrix_sensor_=A,
            whitener_=S,
            dewhitener_=np.linalg.pinv(S),
            mean_=mean,
            alpha_=alpha,
            mu_=mu,
            rho_=rho,
            sbeta_=sbeta,
            c_=c,
            gm_=gm,
            log_likelihood=LL,
            n_iter=len(LL),
            converged=True,
            data_scale=1.0,
        )

        # Reconstruct the Amica object
        config = AmicaConfig(num_mix_comps=n_mix)
        model = cls(config=config)
        model.result_ = result
        return model


def amica(
    X,
    n_components=None,
    whiten=False,
    return_n_iter=False,
    random_state=None,
    max_iter=2000,
    num_mix=3,
    **kwargs,
):
    """Adaptive Mixture ICA (AMICA).

    Compatible with MNE-Python's ``ICA(method='amica')``.
    Follows the Picard integration pattern.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_components)
        Pre-whitened data (MNE convention: samples x components).
    n_components : int or None
        Number of components. If None, uses X.shape[1].
    whiten : bool
        If True, whiten the data. MNE passes False (pre-whitened).
    return_n_iter : bool
        If True, return (W, n_iter) tuple.
    random_state : int or None
        Random seed for reproducibility.
    max_iter : int
        Maximum number of EM iterations.
    num_mix : int
        Number of generalized Gaussian mixture components per source.
    **kwargs
        Additional parameters passed to AmicaConfig.

    Returns
    -------
    W : ndarray, shape (n_components, n_components)
        Unmixing matrix.
    n_iter : int
        Number of iterations (only if return_n_iter=True).
    """
    from .config import AmicaConfig

    # random_state is passed to Amica solver, which uses jax.random.PRNGKey

    # Build config from kwargs
    cfg_kwargs = {
        "max_iter": max_iter,
        "num_mix_comps": num_mix,
        "do_sphere": whiten,
        "do_mean": whiten,
    }
    cfg_kwargs.update(kwargs)
    config = AmicaConfig(**cfg_kwargs)

    # AMICA expects (n_channels, n_samples), MNE passes (n_samples, n_components)
    data = X.T  # (n_components, n_samples)

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data)

    W = result.unmixing_matrix_white_  # (n_components, n_components)

    if return_n_iter:
        return W, result.n_iter
    return W
