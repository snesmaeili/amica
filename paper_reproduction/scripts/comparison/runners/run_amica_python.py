"""Runner for amica-python (Sina's, JAX or NumPy depending on AMICA_NO_JAX).

Reads (n_components, n_samples) from --input. Sina's amica-python expects
data in (n_channels=n_components, n_samples) shape since the orchestrator
already PCA-projected.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_data, parse_runner_args, peak_rss_gb, write_result


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape

    no_jax = os.environ.get("AMICA_NO_JAX", "0") == "1"
    impl = "amica_python_numpy" if no_jax else "amica_python_jax"

    # Import after RSS-baseline measurement
    _ = peak_rss_gb()
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        max_iter=cfg["max_iter"],
        num_mix_comps=cfg.get("n_mix", 3),
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        do_sphere=False,    # already PCA-projected by orchestrator
        do_mean=False,
    )
    model = Amica(config, random_state=cfg.get("seed", 0))

    t0 = time.perf_counter()
    result = model.fit(X)
    elapsed = time.perf_counter() - t0

    W = np.asarray(result.unmixing_matrix_white_)
    ll_history = np.asarray(result.log_likelihood).tolist()

    out = {
        "implementation": impl,
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak_rss_gb(),
        "ll_final": float(ll_history[-1]) if ll_history else float("nan"),
        "ll_history": ll_history,
        "W": W.tolist(),
        "device": "cpu",
        "dtype": "float64",
        "n_iter": int(result.n_iter),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
