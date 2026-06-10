"""Adapter for pyamica (PyTorch implementation)."""

import time

import numpy as np
import torch

from .base import AmicaAdapter


class PyamicaAdapter(AmicaAdapter):

    @property
    def name(self) -> str:
        return "pyamica"

    def run(self, data, params, n_iters, shared_sphere=None,
            shared_mean=None, log_det_sphere=None):
        from pyamica import AMICA

        use_shared = shared_sphere is not None
        n_ch, n_samples = data.shape

        if use_shared:
            # Pre-whiten data externally using shared sphere/mean
            mean = shared_mean if shared_mean is not None else data.mean(axis=1)
            data_white = shared_sphere @ (data - mean[:, None])
            X = torch.from_numpy(data_white.T)  # (T, n_components)
            do_sphere = False
        else:
            X = torch.from_numpy(data.T)  # (T, n_channels)
            do_sphere = True

        model = AMICA(
            n_components=params.get("pcakeep"),
            n_models=1,
            n_mix=params["num_mix"],
            max_iter=n_iters,
            lrate=params["lrate"],
            lrate0=params["lrate"],
            newtrate=params["newtrate"],
            do_newton=True,
            newt_start=params["newt_start"],
            newt_ramp=params["newt_ramp"],
            rho0=params["rho0"],
            minrho=params["minrho"],
            maxrho=params["maxrho"],
            rholrate=params["rholrate"],
            invsigmin=params["invsigmin"],
            invsigmax=params["invsigmax"],
            maxdecs=params["max_decs"],
            min_dll=params["min_dll"],
            use_grad_norm=params.get("use_grad_norm", False),
            use_min_dll=True,
            do_sphere=do_sphere,
            doscaling=params["doscaling"],
            fix_init=True,
            do_reject=params["do_reject"],
            verbose=False,
            dtype=torch.float64,
        )

        t0 = time.perf_counter()
        model.fit(X)
        elapsed = time.perf_counter() - t0

        # Extract and normalize shapes
        # pyamica: (M, n, J) → squeeze M, transpose to (J, n)
        W = model.W_[0].cpu().numpy()
        A = model.A_[0].cpu().numpy()
        alpha = model.alpha_[0].cpu().numpy().T  # (n, J) → (J, n)
        mu = model.mu_[0].cpu().numpy().T
        beta = model.sbeta_[0].cpu().numpy().T
        rho = model.rho_[0].cpu().numpy().T
        c = model.c_[0].cpu().numpy()
        ll = model.LL_.cpu().numpy()

        if use_shared:
            sphere = shared_sphere
            mean_out = shared_mean
            ldet = log_det_sphere if log_det_sphere is not None else 0.0
            # Post-correct LL: pyamica computed sldet from the identity-like
            # covariance of pre-whitened data; replace with real sldet
            wrong_sldet = float(model.sldet_)
            n_comp = W.shape[0]
            ll = ll + (ldet - wrong_sldet) / n_comp
        else:
            # pyamica sphere is (n_orig, n_keep), we need (n_keep, n_orig)
            sphere = model.sphere_.cpu().numpy().T
            mean_out = model.mean_.cpu().numpy()
            ldet = float(model.sldet_)

        return {
            "W": W,
            "A": A,
            "alpha": alpha,
            "mu": mu,
            "beta": beta,
            "rho": rho,
            "c": c,
            "ll_history": ll,
            "sphere": sphere,
            "mean": mean_out,
            "log_det_sphere": ldet,
            "elapsed": elapsed,
            "n_iter": len(ll),
        }
