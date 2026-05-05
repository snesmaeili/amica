"""NumPy-based fallback for JAX functions.

This module provides NumPy implementations when JAX is not available.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

# Allow forcing NumPy fallback via env var
use_jax_env = os.environ.get("AMICA_NO_JAX", "0") != "1"
HAS_JAX = False

if use_jax_env:
    try:
        import jax

        # Enable 64-bit precision by default for scientific accuracy.
        # WARNING: This sets the global JAX context. If `amica-python` is used
        # alongside other JAX packages expecting float32, this may cause conflicts.
        jax.config.update("jax_enable_x64", True)

        # Configure persistent compilation cache
        _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "amica-python", "jax_cache"
        )
        os.makedirs(_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

        import jax.numpy as jnp

        HAS_JAX = True
    except ImportError:
        pass

if not HAS_JAX:
    # Use numpy as fallback
    jnp = np

    # Stub for jax.jit - just return the function unchanged
    class _JaxStub:
        """Dummy JAX module mirroring the subset of the JAX API used by AMICA.

        This allows the rest of the codebase to call `jax.vmap`, `jax.random`, etc.,
        without needing constant `if HAS_JAX:` checks everywhere.
        """

        @staticmethod
        def jit(func: Callable = None, **kwargs) -> Callable:
            """Stub for jax.jit. Returns the function uncompiled."""
            # handle @jax.jit() or @jax.jit(static_argnames=...)
            if func is None:

                def wrapper(f):
                    return f

                return wrapper
            return func

        @staticmethod
        def vmap(func: Callable, *args, **kwargs) -> Callable:
            """Vectorize using numpy - matches JAX vmap behavior for tuple returns.

            WARNING: This stub uses a pure Python loop over the first dimension.
            Unlike true `jax.vmap`, this does not actually vectorize operations
            in C/C++ and will be significantly slower for large batch dimensions.
            """

            def vmapped(*arrays):
                results = [func(*[a[i] for a in arrays]) for i in range(len(arrays[0]))]
                # Handle tuple returns like JAX: return tuple of stacked arrays
                if results and isinstance(results[0], tuple):
                    n_outputs = len(results[0])
                    return tuple(np.array([r[j] for r in results]) for j in range(n_outputs))
                return np.array(results)

            return vmapped

        class random:
            """Stub for jax.random using numpy.random."""

            @staticmethod
            def PRNGKey(seed: int):
                """Create a numpy RandomState acting as a JAX PRNGKey."""
                return np.random.RandomState(seed)

            @staticmethod
            def split(key, num: int = 2):
                if hasattr(key, "randint"):
                    seeds = [key.randint(0, 2**31) for _ in range(num)]
                    return [np.random.RandomState(s) for s in seeds]
                return [np.random.RandomState(i) for i in range(num)]

            @staticmethod
            def normal(key, shape):
                if hasattr(key, "randn"):
                    return key.randn(*shape)
                return np.random.randn(*shape)

        class scipy:
            """Stub for jax.scipy."""

            class special:
                @staticmethod
                def logsumexp(a, axis=None):
                    """Stub for jax.scipy.special.logsumexp using scipy.special."""
                    from scipy.special import logsumexp

                    return logsumexp(a, axis=axis)

        class lax:
            @staticmethod
            def cond(pred, true_fun, false_fun, *operands):
                """Stub for jax.lax.cond.

                WARNING: Evaluates `pred` as a Python bool. In actual JAX, if `pred`
                is a traced array, calling `bool()` will force synchronization and
                defeat JIT compilation. Safe here only because this is the non-JIT fallback.
                """
                if pred:
                    return true_fun(*operands)
                else:
                    return false_fun(*operands)

    jax = _JaxStub()

# Export
__all__ = ["jax", "jnp", "HAS_JAX"]


def get_array_module():
    """Get the appropriate array module (jax.numpy or numpy).

    Returns
    -------
    module : module
        The `jnp` or `np` module depending on JAX availability.
    """
    return jnp


def ensure_numpy(x):
    """Convert array to numpy if it's a JAX array.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    numpy_array : np.ndarray
        NumPy version of the input array.
    """
    if HAS_JAX and hasattr(x, "device"):
        return np.asarray(x)
    return np.asarray(x)


def optional_jit(func: Callable) -> Callable:
    """Decorator that applies jax.jit only if JAX is available.

    Parameters
    ----------
    func : Callable
        Function to potentially JIT compile.

    Returns
    -------
    wrapped : Callable
        JIT-compiled function if JAX is available, otherwise the original function.
    """
    if HAS_JAX:
        return jax.jit(func)
    return func
