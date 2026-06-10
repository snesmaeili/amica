# Contributing to amica-python

Thank you for your interest in contributing to amica-python! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository on GitHub
1. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/amica-python.git
   cd amica-python
   ```
1. Create a virtual environment and install in development mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev,mne,jax]"
   pre-commit install
   ```
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

```bash
python -m pytest tests/ -v
```

### Backend coverage

CI runs **NumPy** (all Python versions) and **JAX-CPU** (Python 3.12 only).
JAX-GPU tests must be run locally before merging anything that touches the
solver or accumulator:

```bash
# JAX-CPU (requires jax installed: pip install -e ".[jax]")
pytest tests/ --backend=cpu -v

# JAX-GPU (requires CUDA + jax[cuda12])
pytest tests/ --backend=gpu -v

# Full suite with slow tests
pytest tests/ --backend=gpu --run-slow -v
```

GPU tests (`--backend=gpu`) are the authoritative correctness check for the
JAX path. The CI NumPy run catches regressions in the shared algorithmic core
but will not catch JAX-specific failures.

### Using Nox

We use `nox` to manage development environments and automate tasks. You can run:

- `nox -s tests`: Run tests across all supported Python versions.
- `nox -s lint`: Run the linters.
- `nox -s docs`: Build the documentation.

Install nox via `pip install nox`.

### Code Style and Linting

- We use `ruff` for code formatting and linting. This is enforced via `pre-commit`.
- Run `pre-commit run --all-files` locally before committing to ensure your code complies.
- Follow PEP 8 conventions where applicable.
- Use type hints for function signatures.
- Add docstrings in NumPy format for all public functions and classes.

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers where applicable (e.g., "Fix #42")

## Types of Contributions

### Bug Reports

Please open an issue on GitHub using the **Bug report template**. Be sure to include:
- A minimal reproducible example
- Expected vs. actual behavior
- Your environment (Python version, OS, JAX version if applicable)

### Feature Requests

Please open an issue on GitHub using the **Feature request template**, describing:
- The use case / motivation
- Proposed API or behavior
- Any relevant references (papers, other implementations)

### Code Contributions

1. Ensure your changes pass all existing tests
2. Add tests for new functionality
3. Run `pre-commit` to ensure code style compliance
4. Update docstrings and documentation as needed
5. Submit a pull request against the `main` branch using the provided **Pull Request template**

### Validation and Benchmarks

We especially welcome contributions that:
- Test AMICA against MATLAB reference outputs on new configurations
- Benchmark on new EEG/MEG datasets
- Compare with other ICA methods (Infomax, FastICA, Picard)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Open an issue or contact Sina Esmaeili at sina.esmaeili@umontreal.ca.
```
