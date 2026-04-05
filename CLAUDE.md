# mne-amica

Native Python/JAX implementation of AMICA (Adaptive Mixture ICA) for MNE-Python. Numerically verified against the reference Fortran AMICA 1.7.

## About the User

Sina Esmaeili — PhD student at Universite de Montreal (Institute of Biomedical Engineering, CoCo Lab under Karim Jerbi). Specializes in M/EEG signal processing and denoising. Author of the mne-denoise package. Contact: sina.esmaeili@umontreal.ca

## Current State (as of 2026-04-04)

### What's Done

**Package (complete, 10/10 tests passing):**
- Full AMICA algorithm: EM with adaptive generalized Gaussian mixture source densities, Newton optimization (quadratic convergence), sample rejection
- 8 modules: solver.py (870 LOC), updates.py (744), pdf.py (346), likelihood.py (269), preprocessing.py (333), backend.py (97), config.py (205), mne_integration.py
- Three APIs: `Amica(config).fit(data)`, `amica(X)` (Picard-compatible), `fit_ica(raw)` (MNE integration)
- Learning rate schedule matches Fortran exactly: natgrad at lrate/2, Newton ramp over 10 iterations
- All log-domain arithmetic, integration-by-parts for Newton (never computes f'')

**MATLAB Parity (verified):**

| Config | LL diff | W corr (mean) | W corr (min) |
|--------|---------|---------------|--------------|
| m=1, Newton | 0.0002% | 1.00000 | 0.99999 |
| m=3, Newton | 0.00008% | 0.99999 | 0.99999 |
| m=3, NatGrad | 0.08% | 0.99985 | 0.99953 |

**Synthetic Validation:**
- Amari index: AMICA 0.008, FastICA 0.010, Infomax 0.195, Picard 0.223
- Parameter sensitivity: kappa >= 30 confirmed, m=3 optimal, seed stable (0.009 +/- 0.001)

**Overleaf Report:**
- Project ID: `69d187ee23884ed437b3b21f`
- Complete technical report with all equations, parity results, parameter sensitivity
- Files: main.tex, references.bib, arxiv.sty

### What Needs To Be Done NOW

**High-density EEG validation on ds004505 (Studnicki 2022, dual-layer table tennis):**

1. Run `validation/run_highdens_validation.py` — this runs AMICA, Picard, Infomax, FastICA on 64-channel scalp EEG
2. Each method: 30 ICA components, 500 iterations
3. ICLabel classification on each decomposition (brain/muscle/eye/other IC counts)
4. Results saved to `validation/results/highdens_validation.json`
5. Update the Overleaf report with the results

### Data Location

The ds004505 dataset was transferred via Globus Connect Personal. You need to find it and update `DS_PATH` in `validation/run_highdens_validation.py`.

Look for it with:
```bash
find ~ -maxdepth 5 -type d -name "ds004505" 2>/dev/null
find /project/ -maxdepth 5 -type d -name "ds004505" 2>/dev/null
find /scratch/ -maxdepth 5 -type d -name "ds004505" 2>/dev/null
```

The data structure should be:
```
ds004505/sourcedata/Merged/sub-01/sub-01_Merged.set
ds004505/sourcedata/Merged/sub-01/sub-01_Merged.fdt
...
```

Each subject has 120 scalp EEG + 120 noise + IMU/EMG channels (~281 total). The script picks 64 scalp EEG channels, filters at 1 Hz, crops to 5 min.

## Environment Setup

```bash
# Load Python module (adjust version for your cluster)
module load python/3.11

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install package + dependencies
pip install -e ".[all]"
pip install mne-icalabel

# Verify
python -c "from mne_amica import Amica; print('OK')"
python -c "from mne_icalabel import label_components; print('ICLabel OK')"
```

### Running the Validation

```bash
# Update DS_PATH in the script first!
# Then run (takes ~30 min for 4 methods x 30 components x 500 iters on 64ch)
python validation/run_highdens_validation.py
```

Or with SLURM:
```bash
#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=amica-validation

module load python/3.11
source .venv/bin/activate
python validation/run_highdens_validation.py
```

## After Validation

1. Check `validation/results/highdens_validation.json` for the results
2. Update the Overleaf report (project `69d187ee23884ed437b3b21f`) with a new subsection in Results showing:
   - Table: method vs brain ICs / muscle ICs / eye ICs / time
   - AMICA should have more brain ICs than other methods (target: reproduce Delorme 2012 ranking)
3. Commit results and push

## Key References

- Palmer et al. (2011) AMICA tech report — primary algorithm reference
- Palmer et al. (2008) ICASSP — Newton method
- Palmer et al. (2006) ICA — super-Gaussian source model
- Delorme et al. (2012) PLoS ONE — AMICA #1 of 22 ICA algorithms
- Frank et al. (2023) IEEE BIBE — optimal parameters (max_iter=2000, m=3, kappa>=30)
- Klug et al. (2024) Sci Rep — rejection: 5-10 iters, 3 SD
- Studnicki et al. (2022) Sensors — ds004505 dataset

## Conventions

- Do NOT add Co-Authored-By Claude lines to git commits
- Figure scripts save to `validation/results/`
- All probability computations in log-domain with logsumexp
- Newton statistics via integration-by-parts (never compute f'')

## Package Quick Reference

```python
from mne_amica import Amica, AmicaConfig, amica, fit_ica

# Standalone
config = AmicaConfig(max_iter=2000, num_mix_comps=3, do_newton=True)
model = Amica(config, random_state=42)
result = model.fit(data)  # data: (n_channels, n_samples)

# MNE integration
ica = fit_ica(raw, n_components=20, max_iter=2000)
ica.get_sources(raw)  # standard MNE ICA object

# Functional (Picard-compatible)
W, n_iter = amica(X, return_n_iter=True)  # X: (n_samples, n_components)
```

### Default Parameters (literature-backed)

| Parameter | Default | Reference |
|-----------|---------|-----------|
| max_iter | 2000 | Frank et al. 2023 |
| num_mix_comps | 3 | Frank et al. 2023 |
| do_newton | True | Palmer et al. 2008 |
| newt_start | 50 | Palmer et al. 2008 |
| rejstart | 2 | Klug et al. 2024 |
| rejint | 3 | Klug et al. 2024 |
| rejsig | 3.0 | Klug et al. 2024 |
