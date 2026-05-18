# amica-python

[![Tests](https://github.com/snesmaeili/amica-python/actions/workflows/tests.yml/badge.svg)](https://github.com/snesmaeili/amica-python/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/snesmaeili/amica-python/branch/main/graph/badge.svg)](https://codecov.io/gh/snesmaeili/amica-python)
[![Python](https://img.shields.io/pypi/pyversions/amica-python.svg)](https://pypi.org/project/amica-python/)
[![Docs](https://github.com/snesmaeili/amica-python/actions/workflows/docs.yml/badge.svg)](https://github.com/snesmaeili/amica-python/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> **Note:** This package is under active validation. The core algorithm works and matches MATLAB AMICA numerically, but the full validation suite and documentation are still in progress.

Python reimplementation of AMICA (Adaptive Mixture ICA), originally a closed-source Fortran binary from UCSD (Palmer et al., 2011). AMICA ranked first among 22 ICA algorithms for EEG in mutual information reduction and source dipolarity (Delorme et al., 2012).

This package provides the full algorithm in Python with optional JAX acceleration and MNE-Python integration.

## Installation

```bash
git clone https://github.com/snesmaeili/amica-python.git
cd amica-python
pip install -e ".[all]"
```

Extras: `jax` (JAX backend), `mne` (MNE-Python integration), `dev` (testing).

## Usage

The fastest way to get started is to look at our fully commented examples in the `examples/` directory:
- [**MNE Integration** (`examples/01_mne_integration.py`)](examples/01_mne_integration.py): Shows how to load EEG data, filter it, and run AMICA using the MNE `fit_ica()` wrapper.
- [**Pure JAX Fitting** (`examples/02_pure_jax_fitting.py`)](examples/02_pure_jax_fitting.py): Shows how to configure the core `AmicaConfig` and run hardware-accelerated fitting directly on numpy arrays.

### Core API Quickstart

```python
from amica_python import Amica, AmicaConfig

config = AmicaConfig(max_iter=2000, num_mix_comps=3)
model = Amica(config, random_state=42)
result = model.fit(data)  # (n_channels, n_samples)
sources = model.transform(data)
```

### MNE-Python Integration

`fit_ica` returns a standard `mne.preprocessing.ICA` object, so all MNE methods work out of the box:

```python
from amica_python import fit_ica

ica = fit_ica(raw, n_components=20, max_iter=2000)
ica.plot_components()
ica.apply(raw)
```

### Picard-compatible Functional API

For drop-in replacement in custom ICA pipelines:

```python
from amica_python import amica

W, n_iter = amica(X, return_n_iter=True)  # X: (n_samples, n_components)
```

## Background

AMICA fits an ICA mixture model where each source has its own mixture of generalized Gaussians with adaptive shape (rho), scale (beta), and location (mu). The shape parameter interpolates between Laplacian (rho=1, super-Gaussian) and Gaussian (rho=2), so it adapts to whatever the data actually looks like rather than assuming a fixed distribution.

Convergence uses natural gradient followed by Newton optimization (Palmer et al., 2008). Optional sample rejection (Klug et al., 2024) downweights outlier time points.

## Validation

Numerical parity with MATLAB AMICA 1.7:

| Configuration                | LL difference | W correlation |
| ---------------------------- | ------------- | ------------- |
| 1 model, Newton              | 0.0002%       | > 0.9999      |
| 3 mixtures, Newton           | 0.00008%      | > 0.9999      |
| 3 mixtures, natural gradient | 0.08%         | > 0.9995      |

Amari index on synthetic Laplacian sources (3 ch, 5000 samples): AMICA 0.008, FastICA 0.010, Infomax 0.195.

## Parameters

Defaults follow Frank et al. (2023) and Klug et al. (2024):

| Parameter       | Default | What it does                           |
| --------------- | ------- | -------------------------------------- |
| `max_iter`      | 2000    | EM iterations                          |
| `num_mix_comps` | 3       | Mixture components per source          |
| `do_newton`     | True    | Newton optimization after iteration 50 |
| `do_reject`     | False   | Outlier sample rejection               |
| `rejsig`        | 3.0     | Rejection threshold in SD              |
| `rho0`          | 1.5     | Initial shape parameter                |

## References

- Palmer, Kreutz-Delgado, Makeig (2011). AMICA: An Adaptive Mixture of Independent Component Analyzers with Shared Components. SCCN Technical Report.
- Palmer, Makeig, Kreutz-Delgado, Rao (2008). Newton Method for the ICA Mixture Model. ICASSP.
- Palmer, Kreutz-Delgado, Makeig (2006). Super-Gaussian Mixture Source Model for ICA. ICA.
- Delorme et al. (2012). Independent EEG Sources Are Dipolar. PLoS ONE.
- Frank et al. (2023). Optimal Parameters for AMICA. IEEE BIBE.
- Klug, Berg, Gramann (2024). Optimizing EEG ICA decomposition. Scientific Reports.

## License

BSD-3-Clause
