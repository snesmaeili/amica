# Test suite TODO

## Done (2026-05-28, branch ci-tests-improvements)

- conftest.py: `--backend` / `--run-slow` options, session fixtures
- Round-trip tolerances tightened to 1e-12 (actuals ~5e-16)
- NaN/Inf input guard in `solver.py` + 5 parametrized tests
- pytest-timeout added to deps; `timeout` marker registered
- Backend parity: JAX-CPU vs NumPy-CPU subprocess test (`@pytest.mark.slow`, cosine sim > 0.99)
- Rejection behavioral: W@A=I (atol 1e-10) + LL improves vs no-reject on contaminated data
- Checkpoint resume: 50+50 ≈ 100 iters (atol 1e-13; JAX JIT FP reorder blocks true bit-exact)
- MNE test strategy: `test_mne_real.py` (real MNE) split from `test_mne_integration.py` (mocked)

## Remaining

### 0. Expose rejection mask in `AmicaResult` — low effort, unlocks test coverage

**What**: Store the final `mask_np` (bool array, shape `(n_samples,)`) in `AmicaResult`
so callers can identify which samples were rejected.

**Where**: `solver.py:1106` — TODO comment already in place.
Add `rejection_mask_: np.ndarray | None = None` field to `AmicaResult` dataclass,
populate it at the end of the rejection block.

**Unlocks**: `test_rejection_behavioral` can then assert that injected spike indices
are flagged (i.e. `result.rejection_mask_[spike_idx]` are False).
Currently that assertion is missing because the mask is not exposed.


### 1. `check_estimator` — blocked, Medium-High effort

**What**: `sklearn.utils.estimator_checks.check_estimator(Amica(...))` — full sklearn API compliance.

**Blocker**: `Amica` does not inherit `sklearn.base.BaseEstimator`.
sklearn 1.8 requires `__sklearn_tags__()` which `BaseEstimator` provides.
`Amica.__init__` takes a `config: AmicaConfig` object, not flat keyword params.
sklearn introspects `__init__` params for `get_params()` / `set_params()` — the config object
pattern is incompatible without refactoring.

**What it would take**:
- Inherit `Amica` from `BaseEstimator`
- Either (a) flatten all `AmicaConfig` fields into `Amica.__init__` kwargs, or
  (b) implement `get_params()` / `set_params()` manually to delegate into `self.config`
- Fix any checks that then fail (e.g. `clone()` behaviour, `set_params` round-trip)

scott-huberty already has this in 12 lines — their `Amica` takes flat params not a config object.

---

### 2. Fortran parity: alpha / mu / sbeta / rho — blocked, needs fixtures

**What**: Compare all fitted parameters (not just W) against Fortran `amica17` output.
scott-huberty does this for W, A, alpha, sbeta, mu, rho, LL.

**Blocker**: No fixture binary files exist in the repo.
The Fortran binary (`amica17`) must be run on a known 6-channel dataset, and its output
directory (files: `W`, `A`, `S`, `mean`, `alpha`, `mu`, `sbeta`, `rho`, `c`, `gm`, `LL`)
committed under `tests/fixtures/fortran_6ch/`.

**What it would take**:
1. Run `amica17` binary on reproducible 6-ch data (fixed seed, saved as `tests/fixtures/fortran_6ch/data.npy`)
2. Commit the output files
3. Write test: load fixture → run Python solver same config → compare all params with tight tolerance
