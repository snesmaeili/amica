# amica

[![CI](https://img.shields.io/github/actions/workflow/status/snesmaeili/amica/tests.yml?branch=main&label=CI)](https://github.com/snesmaeili/amica/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/snesmaeili/amica/docs.yml?branch=main&label=docs)](https://snesmaeili.github.io/amica/)
[![Codecov](https://img.shields.io/codecov/c/github/snesmaeili/amica)](https://codecov.io/gh/snesmaeili/amica)
[![PyPI - Version](https://img.shields.io/pypi/v/amica.svg)](https://pypi.org/project/amica/)
[![Python Versions](https://img.shields.io/pypi/pyversions/amica.svg)](https://pypi.org/project/amica/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> **amica** is a native Python implementation of **AMICA (Adaptive Mixture Independent Component Analysis)**, one of the highest-performing ICA algorithms for EEG source separation.

Originally distributed as a closed-source Fortran executable from UCSD, this implementation of amica provides an open, extensible implementation with optional **JAX acceleration**, seamless **MNE-Python integration**, and a modern Python API for reproducible neuroimaging workflows.

> **Status:** amica is under active development and validation. The implementation already achieves numerical agreement with the original MATLAB AMICA across a broad range of configurations, while documentation and benchmarking continue to expand.

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
git clone https://github.com/snesmaeili/amica.git
cd amica
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
git clone https://github.com/snesmaeili/amica.git
cd amica

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
from amica import Amica, AmicaConfig

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
from amica import fit_ica

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

**https://snesmaeili.github.io/amica/**

______________________________________________________________________

# Validation

amica has been validated against the original MATLAB AMICA implementation and achieves numerical agreement across a wide range of configurations.

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

If amica contributes to your research, please cite the original AMICA publications.

Citation metadata is available in
[CITATION.cff](CITATION.cff).

______________________________________________________________________

# License

amica is distributed under the terms of the BSD 3-Clause License.
