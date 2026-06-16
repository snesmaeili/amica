"""Fortran AMICA 1.7 density-parameter parity (6-channel synthetic fixture).

Runs amica-python from the committed Fortran reference's exact initialisation (mean + sphere +
fix_init) and checks that the unmixing matrix, sources, log-likelihood, AND the adaptive
generalized-Gaussian density parameters (alpha / mu / sbeta / rho) agree to high precision.

The fixture (tests/fixtures/fortran_6ch/) holds the float32 data.fdt and the binary outputs of the
validated amica17 (-DMKL) reference build. Slow (a full 2000-iteration fit) — run with
`pytest --run-slow`. Reference values: tests/fixtures/fortran_6ch/parity_reference.json.
"""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "fortran_6ch"


@pytest.mark.slow
def test_fortran_density_parity(tmp_path):
    if not (FIXTURE / "out" / "alpha").exists():
        pytest.skip("Fortran 6-ch fixture not present")
    # copy to a writable workdir (cmd_compare writes parity.json into it)
    wd = tmp_path / "fortran_6ch"
    shutil.copytree(FIXTURE, wd)
    from amica_python.benchmark.parity.run_fortran_parity import cmd_compare

    out = cmd_compare(SimpleNamespace(workdir=str(wd)))

    assert out["python_n_iter"] == out["fortran_n_iter"] == 2000
    assert out["abs_ll_delta"] < 1e-6, out["abs_ll_delta"]
    assert out["W_matched_abs_r_mean"] > 0.9999, out["W_matched_abs_r_mean"]
    assert out["matched_source_abs_r_mean"] > 0.9999, out["matched_source_abs_r_mean"]
    # the adaptive density parameters that distinguish AMICA
    assert out["alpha_matched_abs_r"] > 0.999, out["alpha_matched_abs_r"]
    assert out["mu_matched_abs_r"] > 0.999, out["mu_matched_abs_r"]
    assert out["sbeta_matched_abs_r"] > 0.999, out["sbeta_matched_abs_r"]
    assert out["rho_matched_abs_r"] > 0.999, out["rho_matched_abs_r"]
