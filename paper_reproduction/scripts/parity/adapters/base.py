"""Abstract base class for AMICA implementation adapters."""

from abc import ABC, abstractmethod

import numpy as np


class AmicaAdapter(ABC):
    """Uniform interface for any AMICA implementation.

    All shapes are normalized to:
      - data: (n_channels, n_samples) — channels first
      - W, A: (n_components, n_components) — whitened space
      - alpha, mu, beta, rho: (n_mix, n_components)
      - sphere: (n_components, n_channels)
      - c: (n_components,)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(
        self,
        data: np.ndarray,
        params: dict,
        n_iters: int,
        shared_sphere: np.ndarray | None = None,
        shared_mean: np.ndarray | None = None,
        log_det_sphere: float | None = None,
    ) -> dict:
        """Run AMICA for n_iters iterations.

        Parameters
        ----------
        data : (n_channels, n_samples) float64 array
        params : aligned parameter dict
        n_iters : number of iterations
        shared_sphere : if provided, use this sphere instead of computing own
        shared_mean : if provided, use this mean
        log_det_sphere : log-determinant of the shared sphere

        Returns
        -------
        dict with keys:
            W : (n, n) unmixing matrix in whitened space
            A : (n, n) mixing matrix in whitened space
            alpha : (J, n) mixture weights
            mu : (J, n) mixture means
            beta : (J, n) inverse scales
            rho : (J, n) shape parameters
            c : (n,) center
            ll_history : (n_iters,) log-likelihood per iteration
            sphere : (n, n_ch) sphering matrix used
            mean : (n_ch,) data mean used
            log_det_sphere : float
            elapsed : float, seconds
        """
        ...
