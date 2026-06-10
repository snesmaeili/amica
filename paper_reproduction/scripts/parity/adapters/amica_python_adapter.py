"""Adapter for amica-python (JAX implementation)."""

import time

import numpy as np

from .base import AmicaAdapter


class AmicaPythonAdapter(AmicaAdapter):

    @property
    def name(self) -> str:
        return "amica-python"

    def run(self, data, params, n_iters, shared_sphere=None,
            shared_mean=None, log_det_sphere=None):
        from amica_python import Amica, AmicaConfig

        cfg = AmicaConfig(
            num_models=1,
            num_mix_comps=params["num_mix"],
            max_iter=n_iters,
            pcakeep=params.get("pcakeep"),
            lrate=params["lrate"],
            newtrate=params["newtrate"],
            newt_start=params["newt_start"],
            newt_ramp=params["newt_ramp"],
            do_newton=True,
            rho0=params["rho0"],
            minrho=params["minrho"],
            maxrho=params["maxrho"],
            rholrate=params["rholrate"],
            invsigmin=params["invsigmin"],
            invsigmax=params["invsigmax"],
            max_decs=params["max_decs"],
            min_dll=params["min_dll"],
            do_reject=params["do_reject"],
            doscaling=params["doscaling"],
            sphere_type=params.get("sphere_type", "pca"),
            use_grad_norm=params.get("use_grad_norm", False),
        )

        m = Amica(cfg, random_state=42)
        t0 = time.perf_counter()

        kwargs = {}
        if shared_sphere is not None:
            kwargs["init_sphere"] = shared_sphere
        if shared_mean is not None:
            kwargs["init_mean"] = shared_mean

        res = m.fit(data, **kwargs)
        elapsed = time.perf_counter() - t0

        return {
            "W": np.asarray(res.unmixing_matrix_white_),
            "A": np.asarray(res.mixing_matrix_white_),
            "alpha": np.asarray(res.alpha_),
            "mu": np.asarray(res.mu_),
            "beta": np.asarray(res.sbeta_),
            "rho": np.asarray(res.rho_),
            "c": np.asarray(res.c_),
            "ll_history": np.asarray(res.log_likelihood),
            "sphere": np.asarray(res.whitener_),
            "mean": np.asarray(res.mean_),
            "log_det_sphere": float(getattr(m, "log_det_sphere", 0.0)),
            "elapsed": elapsed,
            "n_iter": res.n_iter,
        }
