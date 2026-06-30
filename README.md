# PyAMICA

[![CI](https://img.shields.io/github/actions/workflow/status/BabaSanfour/PyAMICA/ci.yml?branch=main&label=CI)](https://github.com/BabaSanfour/PyAMICA/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/BabaSanfour/PyAMICA/docs.yml?branch=main&label=docs)](https://babasanfour.github.io/PyAMICA/)
[![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/PyAMICA)](https://codecov.io/gh/BabaSanfour/PyAMICA)
[![PyPI - Version](https://img.shields.io/pypi/v/pyamica.svg)](https://pypi.org/project/pyamica/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyamica.svg)](https://pypi.org/project/pyamica/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> **PyAMICA** is a native Python implementation of **AMICA (Adaptive Mixture Independent Component Analysis)**, one of the highest-performing ICA algorithms for EEG source separation.

Originally distributed as a closed-source Fortran executable from UCSD, PyAMICA provides an open, extensible implementation with optional **JAX acceleration**, seamless **MNE-Python integration**, and a modern Python API for reproducible neuroimaging workflows.

> **Status:** PyAMICA is under active development and validation. The implementation already achieves numerical agreement with the original MATLAB AMICA across a broad range of configurations, while documentation and benchmarking continue to expand.

______________________________________________________________________

# Highlights

- Native Python implementation of the AMICA algorithm
- Numerical agreement with the original MATLAB AMICA implementation
- Optional **JAX** backend for CPU and GPU acceleration
- Native integration with **MNE-Python**
- Support for **multi-model AMICA**
- Modern scientific Python API
- Extensive testing and continuous integration
- Fully open source (BSD-3-Clause)

______________________________________________________________________

# Installation

## Using pip

```bash
git clone https://github.com/BabaSanfour/PyAMICA.git
cd PyAMICA
pip install -e .
```

Install optional dependencies:

```bash
pip install -e ".[jax]"
pip install -e ".[mne]"
pip install -e ".[dev]"
pip install -e ".[all]"
```

## Using uv

```bash
git clone https://github.com/BabaSanfour/PyAMICA.git
cd PyAMICA

uv venv
source .venv/bin/activate

uv pip install -e .
```

or

```bash
uv pip install -e ".[all]"
```

______________________________________________________________________

# Quick Start

```python
from py_amica import Amica, AmicaConfig

config = AmicaConfig(
    max_iter=2000,
    num_mix_comps=3,
)

model = Amica(config, random_state=42)

result = model.fit(data)

sources = model.transform(data)
```

For MNE-Python:

```python
from py_amica import fit_ica

ica = fit_ica(raw)

ica.plot_components()
ica.apply(raw)
```

______________________________________________________________________

# Examples

Example scripts are available in the `examples/` directory, including:

- MNE-Python integration
- Native AMICA API
- JAX acceleration
- Multi-model AMICA
- HPC / SLURM execution

______________________________________________________________________

# Documentation

Full documentation, API reference, validation experiments, and tutorials are available at

**https://babasanfour.github.io/PyAMICA/**

______________________________________________________________________

# Validation

PyAMICA has been validated against the original MATLAB AMICA implementation and achieves numerical agreement across a wide range of configurations.

The documentation contains:

- validation experiments
- numerical parity analyses
- performance benchmarks
- reproducibility instructions

______________________________________________________________________

# Contributing

Contributions are welcome!

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

______________________________________________________________________

# Citation

If PyAMICA contributes to your research, please cite the original AMICA publications.

Citation metadata is available in
[CITATION.cff](CITATION.cff).

______________________________________________________________________

# License

PyAMICA is distributed under the terms of the BSD 3-Clause License.
