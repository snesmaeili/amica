"""MNE-Python integration for AMICA.

Provides helper functions to use AMICA with MNE-Python's ICA workflow.

Usage
-----
>>> from amica_python import fit_ica
>>> ica = fit_ica(raw, n_components=20, max_iter=2000)
>>> ica.plot_components()
>>> ica.apply(raw)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _extract_data(inst, picks):
    """Extract (n_channels, n_samples) data array from Raw or Epochs.

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        The MNE data object to extract data from.
    picks : array-like of int
        Indices of the channels to extract.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_samples)
        The extracted continuous data. Epochs are concatenated along the time axis.
    """
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw

    if isinstance(inst, BaseRaw):
        return inst.get_data(picks)
    elif isinstance(inst, BaseEpochs):
        # Concatenate all epochs along time axis
        return np.concatenate(inst.get_data()[:, picks, :], axis=-1)
    else:
        raise TypeError(f"inst must be Raw or Epochs, got {type(inst)}")


def _compute_pre_whitener(data, info, picks):
    """Compute MNE-style pre-whitener: per-channel-type std normalization.

    This replicates what MNE does in ICA._pre_whiten when noise_cov=None:
    divide each channel by the std of channels of that type.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        The data to compute the pre-whitener for.
    info : mne.Info
        The measurement info from the MNE object.
    picks : array-like of int
        Indices of the channels corresponding to the rows of `data`.

    Returns
    -------
    pre_whitener : np.ndarray, shape (n_channels, 1)
        The pre-whitening multiplier for each channel.
    """
    from mne import channel_type

    ch_types = [channel_type(info, idx) for idx in picks]
    unique_types = set(ch_types)

    pre_whitener = np.ones((len(picks), 1), dtype=np.float64)
    for ch_type in unique_types:
        mask = np.array([t == ch_type for t in ch_types])
        if mask.sum() > 0:
            std = np.std(data[mask])
            if std > 0:
                pre_whitener[mask, 0] = std

    return pre_whitener


def _compute_pca(data, n_components):
    """Compute PCA on pre-whitened, centered data.

    Computes a full SVD on the data and returns the principal components,
    mean, and explained variance for the full set of components (not truncated).

    Parameters
    ----------
    data : np.ndarray, shape (n_features, n_samples)
        The input data (channels by time).
    n_components : int | None
        Target number of components. Included for API compatibility.

    Returns
    -------
    pca_components : np.ndarray, shape (n_features, n_features)
        The principal components (V^T from SVD).
    pca_mean : np.ndarray, shape (n_features,)
        The mean of the data across samples.
    pca_explained_variance : np.ndarray, shape (n_features,)
        The variance explained by each component.
    """
    import scipy.linalg

    # data is (n_features, n_samples)
    data = data.T  # (n_samples, n_features)
    n_samples = data.shape[0]

    pca_mean = np.mean(data, axis=0)
    data_centered = data - pca_mean

    # Compute full SVD. For PCA we need U, S, Vh
    U, S, Vh = scipy.linalg.svd(data_centered, full_matrices=False)

    pca_components = Vh  # (n_features, n_features)
    pca_explained_variance = (S**2) / (n_samples - 1)

    return pca_components, pca_mean, pca_explained_variance


def fit_ica(
    inst,
    n_components: int | None = None,
    max_iter: int = 2000,
    num_mix: int = 3,
    random_state: int | None = None,
    picks=None,
    reject=None,
    flat=None,
    decim=None,
    fit_params: dict | None = None,
    verbose=None,
):
    """Fit ICA using AMICA on MNE Raw or Epochs data.

    This function replicates MNE's whitening/PCA pipeline, then runs
    AMICA for the unmixing step. The result is a standard MNE ICA object
    that works with all MNE ICA methods (plot_components, apply, etc.).

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        MNE data object.
    n_components : int | None
        Number of ICA components. If None, equals n_channels.
    max_iter : int
        Maximum AMICA iterations. Default 2000.
    num_mix : int
        Number of generalized Gaussian mixture components. Default 3.
    random_state : int | None
        Random seed.
    picks : str | array-like | None
        Channels to use for ICA.
    reject : dict | None
        Epoch rejection parameters.
    flat : dict | None
        Flat channel rejection parameters.
    decim : int | None
        Decimation factor.
    fit_params : dict | None
        Additional parameters passed to AmicaConfig.
    verbose : bool | None
        Verbosity.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with AMICA decomposition.

    Examples
    --------
    >>> from amica_python import fit_ica
    >>> ica = fit_ica(raw, n_components=20)
    >>> ica.plot_sources(raw)
    >>> ica.apply(raw)
    """
    try:
        from mne.preprocessing import ICA
    except ImportError as err:
        raise ImportError(
            "MNE-Python is required for fit_ica(). Install with: pip install mne"
        ) from err

    from amica_python import Amica, AmicaConfig

    # Guard: MNE's ICA assumes a single decomposition matrix.
    _fit_params = fit_params or {}
    _num_models = _fit_params.get("num_models", 1)
    if _num_models > 1:
        raise ValueError(
            f"fit_ica() only supports single-model AMICA (num_models=1), "
            f"got num_models={_num_models}. For multi-model AMICA, use "
            f"Amica(config).fit(data) directly and access AmicaResult."
        )

    # ================================================================
    # Direct MNE ICA construction (no throwaway Infomax)
    # ================================================================
    import mne

    # Resolve picks using public MNE API
    if picks is None:
        # Default: all data channels (eeg, meg, etc.) excluding bads — same as MNE ICA
        picks_idx = mne.pick_types(inst.info, meg=True, eeg=True, ref_meg=False, exclude="bads")
    elif isinstance(picks, str):
        picks_idx = mne.pick_types(inst.info, **{picks: True}, exclude="bads")
    else:
        picks_idx = np.asarray(picks, dtype=int)

    # Extract data, applying reject/flat if provided for Raw
    from mne.io import BaseRaw as _BaseRaw

    if isinstance(inst, _BaseRaw) and (reject is not None or flat is not None):
        # Create fixed-length epochs to apply rejection (matches MNE ICA behavior)
        events = mne.make_fixed_length_events(inst, duration=1.0)
        epochs = mne.Epochs(
            inst,
            events,
            tmin=0,
            tmax=1.0 - 1.0 / inst.info["sfreq"],
            picks=picks_idx,
            reject=reject,
            flat=flat,
            baseline=None,
            preload=True,
            verbose=verbose,
        )
        raw_data = np.concatenate(epochs.get_data(), axis=-1)
        # picks_idx already applied inside Epochs
        raw_data = raw_data.reshape(len(picks_idx), -1) if raw_data.ndim == 3 else raw_data
    else:
        raw_data = _extract_data(inst, picks_idx)

    n_channels, n_samples = raw_data.shape

    # Resolve n_components
    if n_components is None:
        n_comp = n_channels
    else:
        n_comp = min(n_components, n_channels)

    # Decimation
    if decim is not None and decim > 1:
        import scipy.signal

        logger.info("Decimating data by factor %d using FIR anti-aliasing filter.", decim)
        raw_data = scipy.signal.decimate(raw_data, decim, axis=-1, ftype="fir")
        n_samples = raw_data.shape[1]

    # Step 1: Pre-whiten (per-channel-type std normalization)
    pre_whitener = _compute_pre_whitener(raw_data, inst.info, picks_idx)
    data_pre = raw_data / pre_whitener

    # Step 2: PCA
    pca_components, pca_mean, pca_explained_variance = _compute_pca(data_pre, n_comp)

    # Step 3: Project to PCA space (truncated to n_components)
    data_centered = data_pre - pca_mean[:, None]
    pca_data = pca_components[:n_comp] @ data_centered
    # pca_data shape: (n_comp, n_samples)

    # Normalize to unit variance per component to stabilize AMICA's gradient.
    comp_stds = np.std(pca_data, axis=1, keepdims=True)
    comp_stds[comp_stds == 0] = 1.0
    data_for_amica = pca_data / comp_stds

    # Step 4: Run AMICA
    cfg_kwargs = dict(
        max_iter=max_iter,
        num_mix_comps=num_mix,
        do_sphere=False,
        do_mean=False,
    )
    if fit_params:
        cfg_kwargs.update(fit_params)
    config = AmicaConfig(**cfg_kwargs)  # type: ignore[arg-type]

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data_for_amica)

    W = result.unmixing_matrix_white_  # (n_comp, n_comp), operates on data_for_amica

    # Step 5: undo per-component unit-variance normalisation applied at line ~269.
    # AMICA was fed pca_data / comp_stds; recovering the unmixer for unwhitened
    # pca_data requires dividing each column j of W by comp_stds[j].
    # MNE's _transform computes sources = unmixing_matrix_ @ pca_components_ @ centered_data
    # i.e. unmixing_matrix_ must operate on unwhitened X_pca = pca_components_ @ centered_data
    # (the sqrt(eigvals) factor is *not* applied here — MNE bakes it into
    # unmixing_matrix_ at fit time only for backends that receive whitened input,
    # which AMICA does not).
    W_corrected = W / comp_stds.squeeze()[np.newaxis, :]

    # Step 6: Construct MNE ICA object with all required attributes
    # MNE validates method at __init__ — use 'infomax' placeholder, override below
    ica = ICA(n_components=n_comp, method="infomax", max_iter=max_iter)

    # Channel info
    ica.info = mne.pick_info(inst.info, picks_idx)
    ica.ch_names = [inst.info["ch_names"][i] for i in picks_idx]

    # Pre-whitening
    ica.pre_whitener_ = pre_whitener

    # PCA
    ica.pca_components_ = pca_components
    ica.pca_mean_ = pca_mean
    ica.pca_explained_variance_ = pca_explained_variance

    # ICA decomposition (W_corrected operates on unwhitened X_pca — same convention
    # as MNE's stored unmixing_matrix_ for picard/fastica/infomax post-fit).
    ica.n_components_ = n_comp
    ica.unmixing_matrix_ = W_corrected
    ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

    # Metadata
    ica.n_iter_ = result.n_iter
    ica.n_samples_ = n_samples
    ica.current_fit = "raw" if isinstance(inst, _BaseRaw) else "epochs"
    ica.method = "amica"
    ica.labels_ = dict()
    ica.exclude = []
    ica.reject_ = reject
    ica.drop_inds_ = np.array([], dtype=int)

    # Internal naming
    try:
        ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_comp)]
    except Exception:
        pass

    # Attach full AMICA result for viz module
    ica.amica_result_ = result

    return ica
