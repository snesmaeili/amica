# AMICA-Python benchmark — paper-grade formula + figure audit

**Result set:** `v3_paper_stage1_cluster/` (25 subjects × 5 methods, ds004505 TableTennis, 120 ch → 64 PCA, 250 Hz, 52 min).
**Scope:** verify formulas (MIR, PMI, remnant PMI, κ, dipolarity) and the 7 paper figures against Frank 2022, Frank 2023, Frank 2025, Delorme 2012, and Palmer 2011. Documentation-only — no code edits.

Reference figures rendered from the source PDFs are stored in [refs/](refs/) (see [refs/README](refs/) note at bottom of this file).

---

## 1. Summary table

| Quantity / Figure | Status | One-line verdict |
|---|---|---|
| MIR formula | **PASS** | Matches Frank 2022 eq. (7) and Frank 2025 eq. (1); slogdet-stable; gauge-corrected. Citation in docstring says "eq. 5" — should say eq. (7). |
| PMI / mean pairwise MI | **PASS** | Matches Frank 2022 eq. (6) (and the bivariate I form, eq. (1)); symmetric; diag=0; nonneg. |
| Remnant PMI | **PASS (formula + value, diagnosed)** | Formula matches Frank 2022 §II.B. Diagnostic in §2.3 confirms `pmi_input` is byte-identical across all 5 methods per subject (std = 0.0 over 25 subjects), so the normalisation is correct. The AMICA-higher result is a REAL algorithm property: AMICA's mixture-of-Gaussians objective is not equivalent to minimising pairwise MI; paired test shows AMICA > comparator in 88-100% of subjects. |
| κ data-sufficiency | **PASS** | `n_samples / n²`; both κ_channels and κ_effective populated in JSON `_data` block. |
| Dipolarity 4-shell BEM | **PASS** | Radii 71/72/79/85 mm, σ = 0.33 / 0.0042 / 1.0 / 0.33 S/m — exactly Frank 2022 §II.B. |
| Fig 01 — Cumulative dipolarity | **paper-ready** | Matches Delorme 2012 Fig 4A / Frank 2022 Fig 1 structure. Low cumulative-dipolarity is dataset-driven (TableTennis movement) — flag in body text. |
| Fig 02 — 4-panel quality summary | **cosmetic fix** | Caption says "each point is the across-subject method centroid" — true for panels A/C but stale for B/D (which now use ~125 pooled subject×method points per commit f9f80087f0). |
| Fig 04 — MIR difference | **paper-ready** | Sign convention (AMICA advantage = positive) is consistent with Frank 2022 Fig 4. |
| Fig 05 — Runtime | **paper-ready** | Log-y bar chart, AMICA-NumPy positioned correctly as slowest; framing is honest. |
| Fig 07 — AMICA convergence | **structural fix** | Caption and code list iteration milestone **"3000"** that has no source in Frank 2023. The paper uses 50/250/1000/**2000 (default stop)**/**5000 (max)**. |
| Fig 08 — κ data sufficiency | **paper-ready** | Thresholds 20/30/50 match Delorme 2012 / Frank 2025; "no clear plateau past κ=50" phrasing is directly supported by Frank 2025 ("no definitive plateau was observed across the tested range"). |
| Fig 09 — Paired ΔMIR | **paper-ready** | Cohen's d_z = mean_diff/sd_diff is the correct paired-difference effect size; 95% CI half-width 1.96·SE is standard. Per-subject paired t-test correctly applied. |

---

## 2. Formula correctness

### 2.1 MIR (Frank 2022 eq. 7 / Frank 2025 eq. 1)

**Paper:** Frank 2022 eq. (7) ([refs/frank2022_fig1_cumulative_dipolarity.png](refs/frank2022_fig1_cumulative_dipolarity.png) → page-2 equations):

```
MIR = I(x) − I(y)
    = log|det W| + [h(x_1)+...+h(x_n)] − [h(y_1)+...+h(y_n)]                           (7)
```

Frank 2025 restates the same identity as eq. (1) ([refs/frank2025_fig1_data_frames_vs_kappa.png](refs/frank2025_fig1_data_frames_vs_kappa.png)).

**Implementation:** [metrics.py:229–330](../../../../../amica_python/benchmark/metrics.py)

- `log|det W|` via `np.linalg.slogdet` ([metrics.py:306–309](../../../../../amica_python/benchmark/metrics.py)):
  ```python
  sign, logabsdet = np.linalg.slogdet(np.asarray(W_square, dtype=float))
  if sign == 0:
      raise ValueError("W is singular (det = 0); cannot compute MIR.")
  log2_abs_det_W = float(logabsdet / np.log(2.0))
  ```
  Numerically stable (avoids `log(abs(det))` underflow/overflow); converts nat→bit via `1/ln(2)`.

- Entropy sums use identical histogram estimator on input and sources with the same `n_bins`, `clip_sd` ([metrics.py:316–317](../../../../../amica_python/benchmark/metrics.py)):
  ```python
  h_input   = float(sum(entropy_histogram(X_sub[i], n_bins=n_bins, clip_sd=clip_sd) for i in range(X_sub.shape[0])))
  h_sources = float(sum(entropy_histogram(Y_sub[i], n_bins=n_bins, clip_sd=clip_sd) for i in range(Y_sub.shape[0])))
  ```

- Unit conversion ([metrics.py:321](../../../../../amica_python/benchmark/metrics.py)):
  ```python
  kbits_per_sec=float(bits_per_sample * float(sfreq_hz) / 1000.0),
  ```

- Subspace guard ([metrics.py:200–226](../../../../../amica_python/benchmark/metrics.py)) raises if `W` is rectangular and `subspace_mode=False`. The `CompleteMIR.subspace_mode` flag is set when the user opts into PCA-rank-space MIR ([metrics.py:328, 495](../../../../../amica_python/benchmark/metrics.py)).

- Gauge fix ([metrics.py:482–485](../../../../../amica_python/benchmark/metrics.py)) — pre-MIR row σ-normalisation of sources with matched W row rescale:
  ```python
  sigma = Y_raw.std(axis=1, keepdims=True)
  sigma = np.where(sigma > 0, sigma, 1.0)
  Y = Y_raw / sigma
  W_square = W_raw / sigma   # row-rescale W so Y = W_square @ X_pca holds
  ```
  Pure row scaling: preserves the identity `Y = W·X` analytically and preserves MIR (the `log|det W|` term picks up `-Σ log σ_i`, exactly offsetting the `Σ h(y_i/σ_i) - Σ h(y_i) = -Σ log σ_i` shift from row scaling). Test [tests/test_benchmark_metrics.py:186–223](../../../../../tests/test_benchmark_metrics.py) pins source-scale invariance numerically.

**Verdict:** **PASS**. Math is correct, numerically stable, gauge-corrected, and protected against rectangular-W misuse.

**Minor citation bug:** [metrics.py:311, 313](../../../../../amica_python/benchmark/metrics.py) reference *"Frank 2022 eq. 5"*. Eq. (5) in Frank 2022 is the joint-histogram entropy expression; the MIR formula is **eq. (7)**. Same docstring should be updated.

**Tests pinning MIR invariants** ([tests/test_benchmark_metrics.py](../../../../../tests/test_benchmark_metrics.py)):
- `test_identity_transform_zero_mir` (line 44) — W=I → MIR≈0
- `test_permutation_and_sign_transform_zero_mir` (line 53) — permutation+sign-flip → MIR≈0
- `test_orthogonal_rotation_zero_mir` (line 64) — orthogonal W → MIR≈0
- `test_complete_mir_invariant_to_source_scale` (line 186) — per-row σ rescale → MIR unchanged
- `test_complete_mir_uses_stable_slogdet` (line 294) — numerical stability
- `test_clip_choice_does_not_flip_ranking` (line 253) — method ordering robust to ±3/5/10σ histogram clip

---

### 2.2 PMI (Frank 2022 eq. 1 / eq. 6)

**Paper:** Frank 2022 eq. (1): `[M]_ij = I(x_i; x_j) = h(x_i) + h(x_j) − h(x_i, x_j)`. Eq. (6) gives the histogram form `M_lj = H_l + H_j − log(B) − H_lj`.

**Implementation:** [metrics.py:106–164](../../../../../amica_python/benchmark/metrics.py)

```python
mi_nat = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
mi_bits = mi_nat / log2
pmi[i, j] = pmi[j, i] = mi_bits
```

2D histogram uses the same `n_bins / clip_sd` regime as the 1D estimator. Diagonal forced to zero; matrix is symmetric by construction.

**Tests** ([tests/test_benchmark_metrics.py](../../../../../tests/test_benchmark_metrics.py)):
- `test_pmi_diagonal_is_zero` (line 77)
- `test_pmi_is_symmetric` (line 84)
- `test_pmi_nonneg_for_independent_gaussians` (line 91)

**Verdict:** **PASS**.

---

### 2.3 Remnant PMI (Frank 2022 §II.B)

**Paper:** ratio of post-ICA mean pairwise MI to pre-ICA mean pairwise MI, expressed as a percentage. Lower is better.

**Implementation:** [metrics.py:333–365](../../../../../amica_python/benchmark/metrics.py):

```python
pmi_input  = mean_pairwise_mi(X_input, ...)
pmi_source = mean_pairwise_mi(Y_sources, ...)
ratio = 100.0 * pmi_source / pmi_input
```

**Verdict (formula + value):** **PASS — diagnosed.**

**Diagnosis (resolved):** The paired pivot in [d:/tmp/diag_remnant_pmi.py](../../../../../../tmp/diag_remnant_pmi.py) (see "diagnostic command" below for verbatim) confirms:

- **`pmi_input_mean_bits` is byte-identical across all 5 methods per subject** (per-subject across-method std = 0.0 exactly over 25 subjects). The denominator of `remnant_pmi = 100 * pmi_source / pmi_input` cancels exactly across methods within a subject — Hypothesis 3 (normalisation bug) is RULED OUT.
- The AMICA-higher result is a **real algorithm property** (Hypothesis 1). Paired per-subject diffs:
  - AMICA NumPy-CPU vs FastICA: +0.00043 bits source PMI, +0.31 pp remnant PMI, 88% of subjects (22/25)
  - AMICA NumPy-CPU vs Infomax: +0.00048 bits, +0.35 pp, 96% of subjects (24/25)
  - AMICA NumPy-CPU vs Picard:  +0.00049 bits, +0.35 pp, 100% of subjects (25/25)
- JAX-GPU and NumPy-CPU AMICA produce identical `pmi_source` to 5 decimal places (0.029784 vs 0.029786) — backend parity reconfirmed.

**Mechanism:** AMICA optimises mixture-of-Gaussians source likelihood, not pairwise decorrelation. The orthogonalised solutions (Picard, Infomax, FastICA) enforce pairwise decorrelation architecturally. AMICA can therefore find decompositions with higher overall MIR (lower joint mutual information across all source dimensions, eq. 7) while leaving slightly more pairwise correlation. Frank 2022 §II.B's assumption that MIR and remnant PMI move strictly together is empirically weak when the algorithm objectives differ.

**Manuscript writing implication:** report the higher MIR as the primary result (paired t-test p < 1e-11, d_z > 2.5) with a footnote: *"AMICA's mixture-model objective is not strictly equivalent to minimising pairwise mutual information; remnant PMI is on average 1.4% relative higher than Picard / Infomax / FastICA in paired tests (n = 25 subjects). See supplementary diagnostic."*

**Diagnostic command** (run from `repos/amica-python/` after activating the venv):

```bash
wsl bash -lc "cd /mnt/d/amica-validation-workspace/repos/amica-python && \
  source /home/sss/venvs/amica/bin/activate && \
  python -c '
import pandas as pd
from pathlib import Path
R = Path(\"scripts/cc_benchmark/results/v3_paper_stage1_cluster\")
df = pd.read_csv(R/\"benchmark_results.csv\")
piv = df.pivot_table(index=\"subject\", columns=\"method\",
                     values=[\"pmi_input_mean_bits\",\"pmi_source_mean_bits\",\"remnant_pmi_percent\"])
print(piv.head(5))
print(\"---\")
print(\"pmi_input per-subject across-method std (should be ~0):\")
print(piv[\"pmi_input_mean_bits\"].std(axis=1).describe())
'"
```

If `pmi_input` is identical across methods per subject → bug ruled out. If not → bug in input metric or in the order-of-operations of histogram construction.

---

### 2.4 κ data-sufficiency

**Paper:** Delorme 2012 (κ = n_samples / n_channels²); Frank 2025 also reports κ_effective = n_samples / n_components².

**Implementation:** [metrics.py:39–47](../../../../../amica_python/benchmark/metrics.py):

```python
return float(n_samples) / float(n * n)
```

**JSON `_data` block** ([runner.py:414–415, 434–437](../../../../../amica_python/benchmark/runner.py)):
```python
kappa_channels  = float(n_samples_used) / float(n_channels_used ** 2)
kappa_effective = float(n_samples_used) / float(n_components_used ** 2)
# ...
"kappa_channels": kappa_channels,
"kappa_effective": kappa_effective,
"kappa_target_minimum": 30,
"kappa_target_paper_grade": 50,
```

**Verdict:** **PASS**.

---

### 2.5 Dipolarity 4-shell BEM (Frank 2022 §II.B)

**Paper:** Frank 2022 §II.B prescribes radii 71/72/79/85 mm and conductivities 0.33 / 0.0042 / 1 / 0.33 S/m for the spherical 4-shell BEM model.

**Implementation:** [dipolarity.py:29–35](../../../../../amica_python/benchmark/dipolarity.py):

```python
SPHERE_FRANK_2022 = dict(
    r0=(0.0, 0.0, 0.0),
    head_radius=0.085,                  # outer scalp radius in metres (85 mm)
    relative_radii=(71/85, 72/85, 79/85, 1.0),
    sigmas=(0.33, 0.0042, 1.0, 0.33),   # brain / skull / CSF / scalp (S/m)
)
```

Radii at 71, 72, 79, 85 mm; conductivities 0.33 / 0.0042 / 1.0 / 0.33 S/m — **exactly Frank 2022 §II.B**.

`mne.fit_dipole` call at [dipolarity.py:183–185](../../../../../amica_python/benchmark/dipolarity.py); per-IC RV% = `100 − gof` recorded in `dipole_residual_variance_percent` column of [component_metrics.csv](component_metrics.csv).

**Verdict:** **PASS**.

---

## 3. Figure verdicts

### Fig 01 — Cumulative dipolarity ([fig01_cumulative_dipolarity.png](fig01_cumulative_dipolarity.png))

Reference: [refs/delorme2012_fig4_cumulative_dipolarity.png](refs/delorme2012_fig4_cumulative_dipolarity.png) (panel A) and [refs/frank2022_fig1_cumulative_dipolarity.png](refs/frank2022_fig1_cumulative_dipolarity.png).

| Check | Verdict |
|---|---|
| Y axis = cumulative % of ICs with RV ≤ x | OK |
| X axis = dipole RV%, log scale 1–100 | OK |
| Reference lines at 5% (Delorme 2012) and 10% (Frank 2022) | OK |
| All 5 methods drawn with distinct colors | OK |
| Caption cites Delorme 2012 + Frank 2022 cutoffs | OK |

**Verdict: paper-ready.**

**Body-text note** (not a figure fix): our cumulative-dipolarity numbers are lower than the references because ds004505 (TableTennis) has substantially more movement artifact than the cohort-EEG datasets used in Delorme 2012 / Frank 2022. State this directly in the manuscript when the figure is first introduced — do not let the reader infer that AMICA fails to recover dipolar sources.

---

### Fig 02 — Quality summary (Frank 2022 fig 2 style) ([fig02_delorme_style_summary.png](fig02_delorme_style_summary.png))

Reference: [refs/frank2022_fig2_quality_summary.png](refs/frank2022_fig2_quality_summary.png).

| Check | Verdict |
|---|---|
| 4 panels A/B/C/D matching Frank 2022 Fig 2 | OK |
| Panel A: MIR vs near-dipolar share, with regression line | OK |
| Panel B: R² of MIR vs near-dipolar share across RV cutoffs | OK — but caption is stale (see below) |
| Panel C: remnant PMI vs near-dipolar share, with regression line | OK |
| Panel D: R² of remnant PMI vs near-dipolar share across RV cutoffs | OK |
| Vertical 5%/10% RV cutoff markers on B and D | OK |

**Verdict: cosmetic fix.**

**Action:** [captions/fig02_delorme_style_summary_caption.txt](captions/fig02_delorme_style_summary_caption.txt) line 4 says *"Each point is the across-subject method centroid for the same input dataset."* This is correct for **panels A and C** (one dot per method = across-subject mean), but panels **B and D** now perform R² regression on ~125 pooled (subject, method) points after commit f9f80087f0 ([paper_figures.py:188–207, 469–480](../../../../../amica_python/benchmark/viz/paper_figures.py)). Rewrite as:

> Panels A and C: each point is the across-subject method centroid. Panels B and D: R² of the corresponding metric vs near-dipolar share, regressed on per-(subject, method) points (n ≈ 125) at each RV cutoff.

---

### Fig 04 — MIR difference ([fig04_mir_difference.png](fig04_mir_difference.png))

Reference: [refs/frank2022_fig4_5_mir_diff_and_runtime.png](refs/frank2022_fig4_5_mir_diff_and_runtime.png) (top panel = Frank 2022 Fig 4).

Frank 2022 Fig 4 plots `(MIR difference from AMICA)` with **AMICA at 0** and other algorithms with positive bars when they are *worse* (i.e. AMICA's advantage). Our fig04 mirrors this with `(best method's MIR − each comparator's MIR)`, the best method sitting at 0 — same sign convention. The secondary axis showing absolute MIR is a useful addition not in Frank 2022.

**Verdict: paper-ready.**

---

### Fig 05 — Runtime ([fig05_runtime.png](fig05_runtime.png))

Reference: [refs/frank2022_fig4_5_mir_diff_and_runtime.png](refs/frank2022_fig4_5_mir_diff_and_runtime.png) (bottom panel = Frank 2022 Fig 5).

Frank 2022 Fig 5 plots avg run time across 5 decompositions per dataset with AMICA, Picard, FastICA, Pearson-ICA on a linear y-axis. Our fig05 uses **log y-axis**, which is appropriate because AMICA NumPy-CPU is 2 orders of magnitude slower than the JAX-GPU and comparator runs. Caption frames this honestly as "engineering benchmark", not a claim of speed superiority.

**Verdict: paper-ready.**

---

### Fig 07 — AMICA convergence ([fig07_amica_iterations.png](fig07_amica_iterations.png))

Reference: [refs/frank2023_fig1_ll_vs_iter.png](refs/frank2023_fig1_ll_vs_iter.png).

| Check | Verdict |
|---|---|
| Panel A: LL per iteration, per-subject lines + median | OK |
| Panel B: Δ LL per iteration | OK (with the caveat that Frank 2023 plots MIR/PMI per iter, not ΔLL — caption already notes this) |
| Vertical lines at iteration milestones | **WRONG — "3000" has no source** |

**Verdict: structural fix.**

**Action:** [paper_figures.py:811](../../../../../amica_python/benchmark/viz/paper_figures.py) hardcodes `(50, 250, 1000, 2000, 3000)`. Frank 2023 §III explicitly cites:
- **50 iterations** — "default value at which AMICA Newton descent begins"
- **250 iterations** — "dashed line at 250 marks a large change in ΔMIR"
- **1000 iterations** — "diminishing decreases in PMI occur around 1,000 iterations"
- **2000 iterations** — "default maximum used in the EEGLAB implementation of AMICA"
- **5000 iterations** — Frank 2023 used "5000-iteration AMICA decompositions" (also referenced in Frank 2025)

The **3000** value is fabricated. Replace with **5000** (one short Edit):

```python
for axv in (50, 250, 1000, 2000, 5000):
```

And update [captions/fig07_amica_iterations_caption.txt](captions/fig07_amica_iterations_caption.txt) line 3 likewise: `"50, 250, 1000, 2000, 3000"` → `"50, 250, 1000, 2000, 5000"`.

---

### Fig 08 — κ data sufficiency ([fig08_kappa_sufficiency.png](fig08_kappa_sufficiency.png))

Reference: [refs/frank2025_fig1_data_frames_vs_kappa.png](refs/frank2025_fig1_data_frames_vs_kappa.png).

| Check | Verdict |
|---|---|
| Per-subject κ_channels (n_samples/n_chan²) and κ_effective (n_samples/n_comp²) bars | OK |
| Reference lines at κ = 20, 30 (Delorme 2012 min), 50 (Frank 2025 high-data) | OK |
| Bar edge color = regime verdict (red < 30, amber 30–50, teal ≥ 50) | OK |
| Caption phrase "no clear plateau past κ=50" supported by Frank 2025 | OK — Frank 2025 says verbatim *"no definitive plateau in these metrics was observed across the tested range"* |

**Verdict: paper-ready.**

**Note (not a fix):** Frank 2025 Fig 1 plots *number of data frames vs κ* across channel counts — a different visualisation choice (requirement curve, not per-subject snapshot). Our fig08 is the per-subject snapshot conditional on this benchmark's fixed (n_channels, n_components, duration). Both are valid; just clarify in the manuscript prose that fig08 is the **observed κ for our cohort**, not the *required* κ as a function of channels.

---

### Fig 09 — Paired ΔMIR ([fig09_paired_mir_difference.png](fig09_paired_mir_difference.png))

No direct paper analogue — this is a per-subject paired diagnostic that the references do not run on small n. It is analogous to Frank 2022 Fig 3 in spirit but uses paired subjects rather than paired datasets.

| Check | Verdict |
|---|---|
| Per-subject paired Δ MIR (AMICA reference − comparator), one dot per subject | OK |
| Across-subject mean drawn as a black bar | OK |
| 95% CI half-width = 1.96·SE | OK ([paper_figures.py:1026](../../../../../amica_python/benchmark/viz/paper_figures.py)) |
| Cohen's d_z = mean_diff / sd_diff | OK ([paper_figures.py:1027](../../../../../amica_python/benchmark/viz/paper_figures.py)) — the correct paired-difference effect size |
| Paired t-test via `scipy.stats.ttest_rel` | OK ([paper_figures.py:1029](../../../../../amica_python/benchmark/viz/paper_figures.py)) |
| Significance stars at p<0.05, 0.01, 0.001 | OK |
| Stats persisted to [fig09_paired_mir_stats.csv](fig09_paired_mir_stats.csv) | OK |

**Verdict: paper-ready.**

Per-comparator stats (from caption): FastICA: ΔMIR = +0.508 kbits/s, t=+12.48, p=5.51e-12, d_z=+2.50. Infomax: +0.474, t=+12.77, p=3.41e-12, d_z=+2.55. Picard: +0.459, t=+12.52, p=5.17e-12, d_z=+2.50. All three reject H₀ at p < 1e-11 with d_z > 2.5 — interpretation is sound *for this benchmark*; do not extrapolate beyond ds004505 in the manuscript.

---

## 4. Absolute-value plausibility

### MIR scale (4.2–4.7 kbits/s on ds004505 vs Frank 2022's 41.9–43.1 on Delorme 71-ch)

Frank 2022 explicitly cautions that *"the produced MIR values can vary widely across datasets"* (§III). Our values are an order of magnitude smaller than Frank 2022 Table I, but a few-multiple of Frank 2022's no-decomposition baseline of 3.79 kbits/s on Delorme.

Confounders that probably account for the ~10× gap:
- **Bandpass**: ours uses 1.0 Hz high-pass; Frank 2022 used 0.1–100 Hz. Higher high-pass removes low-frequency dependencies and lowers achievable MIR.
- **PCA dimensionality**: ours runs in retained 64-component PCA space; Frank 2022 ran on the full 71-channel space without PCA truncation.
- **Sampling rate**: ours 250 Hz; Frank 2022 250 Hz on Delorme but the kbits/s metric is sample-rate-linear.
- **Dataset**: ds004505 TableTennis (active movement) vs Delorme's seated cognitive datasets.

**Recommended sanity check** (not a fix — a future test): compute the no-decomposition MIR floor on ds004505 itself by setting W = identity and measuring `(Σ h(x_i) − h(x_joint))`. If our floor lands in the 3–4 kbits/s range, the dataset×bandpass×PCA-dim explanation is fully consistent. If it lands at ~0.5 kbits/s, there is a metric scale issue to chase.

### Cumulative dipolarity (2–5% at RV ≤ 5% vs Frank's 30%+ for AMICA on Delorme)

ds004505 is the **TableTennis** dataset: subjects play virtual TT in VR. This generates substantial neck, jaw, and ocular muscle activity, which produces non-dipolar ICs (muscle is typically diffuse or multi-source). The depressed cumulative-dipolarity is consistent with the dataset choice, not a bug.

**Verification** (not blocking): inspect 3–5 `_ica.fif` files and confirm the lowest-RV ICs are plausibly cortical (centro-parietal alpha, occipital alpha) while the high-RV ICs show muscle-like topographies (peripheral, broadband). This is a 30-minute manual check, not a code change.

### Remnant PMI anomaly

See §2.3 for the diagnostic command and three-hypothesis triage. Do not claim AMICA "reduces total mutual information *more* than the comparators" in the manuscript until the remnant-PMI anomaly is reconciled — claim the **higher MIR** result only, and footnote the remnant-PMI observation as "to be diagnosed in supplementary".

---

## 5. Gap analysis vs Frank 2022/2023/2025/Delorme 2012

Ranked by impact for a JOSS / software-paper submission:

1. **Frank 2025 Fig 3 — MIR/dipolarity vs κ subsampling (HIGH impact)** ([refs/frank2025_fig1_data_frames_vs_kappa.png](refs/frank2025_fig1_data_frames_vs_kappa.png) shows the methods page; Frank 2025 Fig 3 is on PDF page 5 of refs/). Not implemented. Would require fitting AMICA at multiple data fractions per subject (e.g. 25%, 50%, 75%, 100% of frames) and re-evaluating MIR and dipolarity. **This is the most paper-relevant missing piece** because it directly answers "is 52 min × 25 subjects enough data?" — the core data-sufficiency question.

2. **Frank 2023 Fig 2 — PMI vs iteration (MEDIUM impact)** ([refs/frank2023_fig1_ll_vs_iter.png](refs/frank2023_fig1_ll_vs_iter.png) shows Fig 1; Fig 2 is on the next page). Not implemented. Requires hooking the AMICA fit loop to log W per iteration (or every N iterations) and computing PMI(Y) post-hoc. Useful for justifying our default `max_iter` choice and for showing AMICA converges *before* the 2000-iteration EEGLAB default.

3. **Frank 2023 — seed stability (MEDIUM-LOW impact)**. Not implemented. Fit AMICA from 5–10 random seeds per subject and report MIR/dipolarity variance. Justifies that our headline numbers are not seed-cherry-picked.

4. **Frank 2022 Fig 3 — per-dataset MIR ordering (N/A for now)**. Not relevant until we run on > 1 dataset. **Strong recommendation**: at least add **ds004504 (FaceWashout)** and **ds003194 (Cued-RT)** before JOSS submission so the single-dataset caveat in the abstract becomes "demonstrated across 3 datasets".

5. **No-decomposition MIR floor on ds004505 (LOW impact, easy)**. Helpful for §4 plausibility argument. ~10 lines of code reusing `complete_mir` with `W=np.eye(64)`.

---

## 6. Action plan

In priority order, each is a small, well-localised edit:

### Required (correctness)
1. **Fix fig07 milestone**: edit [paper_figures.py:811](../../../../../amica_python/benchmark/viz/paper_figures.py) — change `(50, 250, 1000, 2000, 3000)` → `(50, 250, 1000, 2000, 5000)`. Update [captions/fig07_amica_iterations_caption.txt](captions/fig07_amica_iterations_caption.txt) line 3 to match. Regenerate fig07.
2. **Fix MIR citation**: edit [metrics.py:311, 313](../../../../../amica_python/benchmark/metrics.py) — `"Frank 2022 eq. 5"` → `"Frank 2022 eq. 7"`. Two occurrences in `det_note` strings.
3. **Fix fig02 caption**: rewrite [captions/fig02_delorme_style_summary_caption.txt](captions/fig02_delorme_style_summary_caption.txt) line 4 to distinguish A/C (centroid) from B/D (per-(subject, method) points). Suggested wording in §3 above. No figure regeneration needed.

### Strongly recommended (interpretation)
4. **Diagnose remnant-PMI anomaly**: run the diagnostic command in §2.3. If `pmi_input` is identical across methods per subject, the formula is fine and the result is a real AMICA property — footnote it. If `pmi_input` differs, chase the bug in `mean_pairwise_mi` histogram setup.
5. **Compute no-decomposition MIR floor on ds004505**: add a one-shot script that loads `_ica.fif` data, computes `complete_mir(W=I, X=X_pca, Y=X_pca)`, reports kbits/s. Compare to our AMICA result.

### For JOSS submission (scope)
6. **Add ≥ 2 more datasets** (e.g. ds004504, ds003194). This is the highest-leverage thing the paper is missing.
7. **Implement Frank 2025 Fig 3** (MIR/dipolarity vs κ subsampling).
8. **Add seed-stability supplementary** (5 seeds per subject for AMICA).

### Verification (one-shot, read-only)
```bash
# Regenerate figures after milestone + caption edits:
wsl bash -lc "cd /mnt/d/amica-validation-workspace/repos/amica-python && \
  source /home/sss/venvs/amica/bin/activate && \
  python -c '
import pandas as pd; from pathlib import Path
from amica_python.benchmark import viz
R = Path(\"scripts/cc_benchmark/results/v3_paper_stage1_cluster\")
bench_df = pd.read_csv(R/\"benchmark_results.csv\")
comp_df  = pd.read_csv(R/\"component_metrics.csv\")
iter_df  = pd.read_csv(R/\"iteration_trace.csv\")
out = R/\"figures/paper\"; cap = out/\"captions\"
viz.plot_amica_convergence(iter_df, out, cap, bench_df=bench_df)
'"

# Run unit tests:
wsl bash -lc "cd /mnt/d/amica-validation-workspace/repos/amica-python && \
  source /home/sss/venvs/amica/bin/activate && \
  pytest tests/test_benchmark_metrics.py tests/test_mne_integration.py -q"
```

---

## References

- **Delorme et al. 2012**, "Independent EEG sources are dipolar", PLOS ONE 7(2):e30135. ([refs/delorme2012_fig4_cumulative_dipolarity.png](refs/delorme2012_fig4_cumulative_dipolarity.png))
- **Frank et al. 2022**, "A Framework to Evaluate Independent Component Analysis applied to EEG signal — testing on the Picard algorithm", IEEE BIBE 2022. ([refs/frank2022_fig1_cumulative_dipolarity.png](refs/frank2022_fig1_cumulative_dipolarity.png), [refs/frank2022_fig2_quality_summary.png](refs/frank2022_fig2_quality_summary.png), [refs/frank2022_fig4_5_mir_diff_and_runtime.png](refs/frank2022_fig4_5_mir_diff_and_runtime.png))
- **Frank et al. 2023**, "An exploration of optimal parameters for efficient blind source separation of EEG recordings using AMICA". ([refs/frank2023_fig1_ll_vs_iter.png](refs/frank2023_fig1_ll_vs_iter.png))
- **Frank et al. 2025**, "Quantifying data requirements for EEG independent Component Analysis using AMICA". ([refs/frank2025_fig1_data_frames_vs_kappa.png](refs/frank2025_fig1_data_frames_vs_kappa.png))
- **Palmer et al. 2011**, "AMICA: An adaptive mixture of independent component analyzers with shared components". (Algorithm spec; not figure-referenced here.)

Reference PNGs in [refs/](refs/) were rendered at 150 DPI from the source PDFs in `C:/Users/s/Downloads/amica_papers/Paperpile files/AMICA/` using `pdftoppm`. They are committed as a one-time snapshot to keep the audit reproducible.
