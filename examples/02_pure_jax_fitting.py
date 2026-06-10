"""Example 02 — Pure-JAX AMICA on a NumPy array (no MNE needed).

Configure the core :class:`~amica_python.AmicaConfig` and run hardware-accelerated
AMICA directly on a ``(n_channels, n_samples)`` array. Use this when your data is
already a plain NumPy array (a custom pipeline, simulated data, non-EEG signals).

Run::

    python examples/02_pure_jax_fitting.py

JAX uses the GPU automatically when available, otherwise the CPU.
"""
from __future__ import annotations

import numpy as np

from amica_python import Amica, AmicaConfig


def make_synthetic_mixture(n_sources: int = 6, n_samples: int = 20_000, seed: int = 0):
    """A simple ICA problem: a random linear mixing of independent Laplacian
    (super-Gaussian) sources. Returns ``X`` of shape ``(n_channels, n_samples)``."""
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(n_sources, n_samples))        # independent sources
    A = rng.standard_normal((n_sources, n_sources))     # mixing matrix
    return (A @ S).astype(np.float64)                   # observed mixtures


def main() -> None:
    X = make_synthetic_mixture()
    print(f"data: {X.shape}  (channels, samples)")

    config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
    model = Amica(config, random_state=42)
    result = model.fit(X)

    sources = model.transform(X)                        # (n_components, n_samples)
    ll = float(np.asarray(result.log_likelihood)[-1])
    print(f"converged in {int(result.n_iter)} iterations; final log-likelihood = {ll:.4f}")
    print(f"recovered sources: {sources.shape}")
    # `result` also exposes the unmixing/mixing matrices and (for num_models>1)
    # the per-model weights `gm_` and posteriors `model_posteriors_`.


if __name__ == "__main__":
    main()
