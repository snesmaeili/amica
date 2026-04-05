"""MNE-Python integration for AMICA.

Provides helper functions to use AMICA with MNE-Python's ICA workflow.

Usage
-----
>>> from mne_amica import fit_ica
>>> ica = fit_ica(raw, n_components=20, max_iter=2000)
>>> ica.plot_components()
>>> ica.apply(raw)
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


def fit_ica(
    inst,
    n_components: Optional[int] = None,
    max_iter: int = 2000,
    num_mix: int = 3,
    random_state: Optional[int] = None,
    picks=None,
    reject=None,
    flat=None,
    decim=None,
    fit_params: Optional[dict] = None,
    verbose=None,
):
    """Fit ICA using AMICA on MNE Raw or Epochs data.

    This function creates an MNE ICA object, performs whitening/PCA
    using MNE's infrastructure, then runs AMICA for the unmixing step.
    The result is a standard MNE ICA object that works with all MNE
    ICA methods (plot_components, apply, etc.).

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
    >>> from mne_amica import fit_ica
    >>> ica = fit_ica(raw, n_components=20)
    >>> ica.plot_sources(raw)
    >>> ica.apply(raw)
    """
    try:
        from mne.preprocessing import ICA
    except ImportError:
        raise ImportError(
            "MNE-Python is required for fit_ica(). "
            "Install with: pip install mne"
        )

    from mne_amica import amica as _amica_func

    # Step 1: Create MNE ICA and let it handle preprocessing
    # We use 'infomax' as a placeholder to get MNE's whitening pipeline
    ica = ICA(
        n_components=n_components,
        method="infomax",
        random_state=random_state,
        max_iter=1,  # We'll replace the unmixing anyway
    )

    # Fit with infomax (1 iteration, just to set up whitening)
    ica.fit(
        inst,
        picks=picks,
        reject=reject,
        flat=flat,
        decim=decim,
        verbose=verbose,
    )

    # Step 2: Extract whitened data and re-run with AMICA
    # MNE stores the whitened PCA-reduced data pipeline:
    #   raw -> picks -> pre_whitener -> pca_components_ -> unmixing_matrix_
    #
    # We need to get data in the whitened space (after PCA)
    # and run AMICA on it.

    from mne.io import BaseRaw
    from mne.epochs import BaseEpochs

    if isinstance(inst, BaseRaw):
        data = ica._transform_raw(inst, 0, None)
    else:
        data = ica._transform_epochs(inst)

    # data shape: (n_components_, n_samples) or (n_epochs*n_times, n_components_)
    # AMICA's amica() expects (n_samples, n_components) following MNE/Picard convention
    if data.ndim == 2 and data.shape[0] == ica.n_components_:
        # (n_components, n_samples) -> transpose
        data_for_amica = data.T
    else:
        data_for_amica = data

    # Build AMICA kwargs
    amica_kwargs = dict(max_iter=max_iter, num_mix=num_mix)
    if fit_params:
        amica_kwargs.update(fit_params)

    # Step 3: Run AMICA
    W, n_iter = _amica_func(
        data_for_amica,
        whiten=False,
        return_n_iter=True,
        random_state=random_state,
        **amica_kwargs,
    )

    # Step 4: Replace unmixing matrix
    # MNE applies whitening normalization: unmixing /= sqrt(pca_explained_variance_)
    # We need to match this convention
    norms = np.sqrt(ica.pca_explained_variance_[: ica.n_components_])
    norms[norms == 0] = 1.0

    ica.unmixing_matrix_ = W / norms
    ica.n_iter_ = n_iter

    # Recompute mixing matrix
    ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

    # Tag as AMICA
    ica.method = "amica"

    return ica
