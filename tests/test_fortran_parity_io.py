"""Round-trip tests for the Fortran AMICA I/O (no binary needed)."""
import os

os.environ.setdefault("AMICA_NO_JAX", "1")
import numpy as np
import pytest

from amica_python.benchmark.parity import fortran_io as fio


def test_fdt_roundtrip(tmp_path):
    X = np.random.RandomState(0).randn(6, 1000)  # (n_channels, n_samples)
    p = fio.write_fdt(X, tmp_path / "data.fdt")
    back = np.fromfile(p, dtype="<f4").reshape((6, 1000), order="F")
    assert np.allclose(back, X.astype("<f4"), atol=1e-6)


def test_param_written(tmp_path):
    p = fio.write_param(tmp_path / "x.param", files="/data/d.fdt", outdir="/out/o/",
                        n_channels=6, n_samples=1000, num_mix_comps=3, do_newton=0,
                        max_iter=42)
    txt = p.read_text()
    assert "data_dim 6" in txt and "field_dim 1000" in txt
    assert "num_mix_comps 3" in txt and "do_newton 0" in txt and "max_iter 42" in txt
    assert "\r" not in txt  # LF only


def test_results_and_init_roundtrip(tmp_path):
    od = tmp_path / "out"
    od.mkdir()
    nc, nm = 6, 3
    W = np.random.RandomState(1).randn(nc, nc)
    W.ravel(order="F").tofile(od / "W")
    np.zeros(nc).tofile(od / "mean")
    np.eye(nc).ravel(order="F").tofile(od / "S")
    np.linalg.pinv(W).ravel(order="F").tofile(od / "A")
    np.zeros((nc, 1)).ravel(order="F").tofile(od / "c")
    np.array([-1.0, -0.9, -0.8, 0.0, 0.0]).tofile(od / "LL")
    for nmf in ("alpha", "sbeta", "mu", "rho"):
        np.random.RandomState(2).randn(nm, nc).ravel(order="F").tofile(od / nmf)
    res = fio.read_fortran_results(od, n_components=nc, n_mixtures=nm)
    assert np.allclose(res["W"], W)
    assert res["alpha"].shape == (nc, nm)
    assert res["n_iter"] == 3  # trailing zeros dropped

    W.ravel(order="F").tofile(od / "Wtmp.bin")
    np.random.RandomState(3).randn(nm, nc).ravel(order="F").tofile(od / "sbetatmp.bin")
    np.random.RandomState(4).randn(nm, nc).ravel(order="F").tofile(od / "mutmp.bin")
    W0, sb0, mu0 = fio.read_initial_weights(od, n_components=nc, n_mixtures=nm)
    assert np.allclose(W0, W) and sb0.shape == (nc, nm) and mu0.shape == (nc, nm)


def test_llt_roundtrip_and_rejected_set(tmp_path):
    """read_fortran_llt parses the sample-major (M+1)-per-sample float64 layout and
    recovers the rejected set as {total == 0} (rejected samples are zeroed)."""
    od = tmp_path / "out"
    od.mkdir()
    n_samples, M = 50, 1
    rng = np.random.RandomState(5)
    total = rng.standard_normal(n_samples)
    rejected_true = np.array([7, 23, 41])
    total[rejected_true] = 0.0  # Fortran reject_data zeroes loglik
    # On disk: sample-major, [modloglik_1..M, total] per sample (M=1: mod == total).
    arr = np.concatenate([total[:, None], total[:, None]], axis=1)  # (n_samples, M+1)
    arr.astype("<f8").ravel().tofile(od / "LLt")

    modloglik, tot = fio.read_fortran_llt(od, n_samples=n_samples, num_models=M)
    assert modloglik.shape == (n_samples, M) and tot.shape == (n_samples,)
    np.testing.assert_allclose(tot, total)
    np.testing.assert_array_equal(np.where(tot == 0.0)[0], rejected_true)
    # Wrong size is caught (guards the recl-unit / n_samples assumption).
    with pytest.raises(ValueError):
        fio.read_fortran_llt(od, n_samples=n_samples + 1, num_models=M)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    test_fdt_roundtrip(d)
    (d / "p").mkdir()
    test_param_written(d / "p")
    (d / "r").mkdir()
    test_results_and_init_roundtrip(d / "r")
    print("F1 fortran_io round-trip: ALL OK")
