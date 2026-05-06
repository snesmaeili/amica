# amica-python: Second-Pass Production Readiness Plan

## Goal
All code-level blockers (items 1–17 in `package_review.md`) are resolved and pushed to `production-ready-amica`. This plan targets everything that remains before PyPI publication and the MNE #13819 PR.

---

## Remaining Open Items

| # | Item | Priority |
|---|------|----------|
| 12 | Enforce **90–95% coverage** | ✅ Done |
| 13 | Add `CHANGELOG.md` — **MNE-style (towncrier)** | 🟡 High |
| 18 | Multi-subject ds004505 ICLabel benchmark — **SLURM array job** on Compute Canada | 🟡 High |
| 19 | Sphinx docs | 🟢 Post-release |
| 21 | Codecov badge in README | ✅ Done |
| 23 | PyPI publish workflow | 🟢 Post-release |

**Focus this sprint: Items 12 (remainder), 13, 18.**

---

## Decisions (answered)

| Question | Answer |
|----------|--------|
| Coverage target | **90–95%** enforced via `--cov-fail-under=90` |
| Changelog format | **MNE-style towncrier** with `changes/` fragment directory |
| Benchmark execution | **SLURM array job** (`--array=1-25`) on Narval/Compute Canada |
| Release version | **0.0.1** (pre-release) |
| Python versions (CI matrix) | **3.10 – 3.14** |
| Backend CI matrix | **JAX + NumPy** via `AMICA_NO_JAX` env var |
| Test pattern | **Flat functions only** (no test classes) |

---

## Sprint 1: Coverage Hardening (Item 12) — Status

### Current coverage snapshot (2026-05-06)

| Module | Coverage | Status |
|--------|----------|--------|
| `__init__.py` | **100%** | ✅ Done |
| `accumulators.py` | **100%** | ✅ Done — `test_accumulators.py` |
| `likelihood.py` | **100%** | ✅ Done — `test_likelihood.py` |
| `pdf.py` | **96%** | ✅ Done — `test_pdf.py` (lines 10-12 = NumPy fallback, covered in CI NumPy matrix) |
| `preprocessing.py` | **100%** | ✅ Done — `test_preprocessing.py` |
| `updates.py` | **99%** | ✅ Done — `test_updates.py` (lines 12-14 = NumPy fallback import, CI matrix) |
| `viz.py` | **100%** | ✅ Done — `test_viz.py` |
| `backend.py` | **36% (local)** | ⚠️ JAX init block (lines 18-38) only runs when JAX installed. CI JAX matrix job covers it. NumPy stub (42-129) covered by `AMICA_NO_JAX=1` matrix job. |
| `config.py` | **100%** | ✅ Done — `test_config.py` |
| `metrics.py` | **100%** | ✅ Done — `test_metrics.py` |
| `mne_integration.py` | **100%** | ✅ Done — `test_mne_integration.py` |
| `solver.py` | **91%** | ✅ Done — `test_solver_paths.py` |
| `binary.py` | **excluded** | ✅ Excluded from coverage target |

**Package total (excl. binary.py): ~92%** → Target: **90–95%**

### Remaining test files to create

#### [NEW] `tests/test_solver_paths.py`
- [x] `solver.py` (Current: 91%, Target: 90%+)
  - *Strategy*: Focus on the `_amica_step` control flow blocks (`do_newton`, `do_mean`, `pcakeep`) using small mock matrices. Test the checkpointing save/load logic explicitly. Bypass deep JAX numeric validation by focusing on shape propagation and boundary logic. Fixed `vmapped` dummy implementation in `backend.py` to correctly map `in_axes` and eliminate `IndexError` during testing.

---

## Sprint 2: CHANGELOG + Release Infra (Item 13)

Follows the **MNE-Python changelog pattern**: `towncrier` + `changes/` fragment directory.

### Changes

#### [NEW] `CHANGELOG.md`
MNE-style header with `<!-- towncrier release notes start -->` marker. First compiled entry is **`0.0.1` (pre-release)**.

#### [NEW] `changes/` directory
Towncrier fragment directory. Fragment types match MNE:
- `changes/<issue>.enh.md` → Enhancements
- `changes/<issue>.bugfix.md` → Bug Fixes
- `changes/<issue>.api.md` → API Changes
- `changes/<issue>.dep.md` → Dependencies
- `changes/<issue>.authors.md` → Authors

#### [MODIFY] `pyproject.toml`
- Set version `0.0.1` (pre-release, Development Status :: 2 - Pre-Alpha)
- Add `[tool.towncrier]` config block
- Add `towncrier>=23.0` to `dev` extra
- Add `--cov-fail-under=90` to `addopts` once coverage target is reached

#### [MODIFY] `.github/workflows/tests.yml`
- [x] Add `--cov-report=xml` and `--cov-fail-under=90`
- [x] Upload `coverage.xml` to Codecov via GitHub Secrets

---

## Sprint 3: Compute Canada Benchmark (Item 18)

**Executed as a SLURM array job on Narval/Béluga.**

### What the MNE community is waiting for
- Multi-subject ICLabel brain/muscle/eye classification on `ds004505` (25 subjects, 118-ch dual-layer EEG)
- Comparison: AMICA vs Picard vs Infomax vs FastICA
- Metrics: ICLabel label distribution, dipole_rv, runtime per subject, peak RSS

### Architecture

```
scripts/cc_benchmark/
├── submit_all.sh          # sbatch --array=1-25 launcher
├── run_one_subject.py     # per-subject pipeline
├── aggregate_results.py   # produces markdown table for MNE issue
└── narval_env.sh          # module load python + venv activation
```

#### `submit_all.sh`
```bash
#!/bin/bash
#SBATCH --job-name=amica_benchmark
#SBATCH --array=1-25
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sub-%a_%j.out
#SBATCH --error=logs/sub-%a_%j.err

source narval_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID
```

#### `run_one_subject.py`
```python
# 1. Load ds004505/sub-XX/eeg/ via MNE-BIDS
# 2. Preprocess: bandpass 1-100 Hz, notch 60 Hz, bad-ch interpolation
# 3. For each method (AMICA, Picard, Infomax, FastICA):
#    - fit_ica() → ICA object
#    - label_components(raw, ica, method="iclabel")
#    - record: brain%, eye%, muscle%, runtime, peak_rss
# 4. Save results/sub-XX.json
```

### Pre-benchmark checklist
1. Confirm `ds004505` path on Narval (`/scratch/$USER/ds004505`)
2. `pip install mne-icalabel mne-bids` in the Narval venv
3. Local dry-run: `python run_one_subject.py --subject 1 --max-iter 20`
4. SLURM dry-run: `sbatch --array=1-2 submit_all.sh`
5. Full run: `sbatch submit_all.sh`

---

## Verification Plan

### Sprint 1 (Coverage)
```bash
pip install -e ".[dev,mne,jax]"
pytest tests/ --cov=amica_python --cov-report=term-missing
# Target: ≥90% total (excl. binary.py)
```

### Sprint 2 (Release Infra)
- `python -m build` succeeds (wheel + sdist)
- `twine check dist/*` passes
- CHANGELOG dates and version are consistent

### Sprint 3 (Benchmark)
- Local dry run: 1 subject, 20 iterations
- SLURM dry run: `--array=1-2`
- Full run: `--array=1-25`
- Results posted to MNE #13819

---

## Proposed Execution Order
1. **Now**: Sprint 1 remainder — `test_config.py`, `test_metrics.py` extensions, `test_mne_integration.py`, `test_solver_paths.py`
2. **After**: Sprint 2 (CHANGELOG + version bump) — ~30min
3. **Compute Canada**: Sprint 3 (benchmark) — requires Narval access
