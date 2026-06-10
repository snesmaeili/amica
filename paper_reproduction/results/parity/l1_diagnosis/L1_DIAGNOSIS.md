# L1 Sphering Diagnosis

**Generated:** 2026-05-01T21:54:54Z
**Dataset:** mne
**Aligned params source:** `scripts/parity/three_way_parity.py::ALIGNED_PARAMS`

## Adapters
- `amica-python`: sphere shape=(30, 59) dtype=float64; mean shape=(59,) dtype=float64; log_det_sphere=357.726
- `pyamica`: sphere shape=(30, 59) dtype=float64; mean shape=(59,) dtype=float64; log_det_sphere=357.726
- `fortran`: sphere shape=(30, 59) dtype=float64; mean shape=(59,) dtype=float64; log_det_sphere=0

## Pairwise diagnostics
### amica-python vs pyamica
- sphere_max_abs_diff: 1.338048605248332e-08
- sphere_frobenius: 4.262758765450267e-08
- sphere_diag_max_abs_diff: 2.764863893389702e-09
- sphere_offdiag_max_abs_diff: 1.338048605248332e-08
- sphere_row_norm_ratio_min: 0.9999999999999974
- sphere_row_norm_ratio_median: 1.0
- sphere_row_norm_ratio_max: 1.0000000000000024
- mean_max_abs_diff: 5.1285197978287716e-23
- mean_l2: 1.2752440012320764e-22
- dtype_match: True
- sphere_dtype_a: float64
- sphere_dtype_b: float64

### amica-python vs fortran
- sphere_max_abs_diff: 267491.14683476015
- sphere_frobenius: 1574194.5817845173
- sphere_diag_max_abs_diff: 267491.14683476015
- sphere_offdiag_max_abs_diff: 204427.47015797987
- sphere_row_norm_ratio_min: 0.21855931415963448
- sphere_row_norm_ratio_median: 0.9066759540496767
- sphere_row_norm_ratio_max: 2.0652579958179813
- mean_max_abs_diff: 1.089603723096841e-15
- mean_l2: 3.400694597590162e-15
- dtype_match: True
- sphere_dtype_a: float64
- sphere_dtype_b: float64

### pyamica vs fortran
- sphere_max_abs_diff: 267491.1468347589
- sphere_frobenius: 1574194.581784518
- sphere_diag_max_abs_diff: 267491.1468347589
- sphere_offdiag_max_abs_diff: 204427.4701579806
- sphere_row_norm_ratio_min: 0.21855931415963453
- sphere_row_norm_ratio_median: 0.9066759540496765
- sphere_row_norm_ratio_max: 2.0652579958179778
- mean_max_abs_diff: 1.089603729714286e-15
- mean_l2: 3.400694592121834e-15
- dtype_match: True
- sphere_dtype_a: float64
- sphere_dtype_b: float64

## Ranked suspects
1. row-norm scale ratio in pyamica vs fortran (median=0.9067; expected 1.0)
2. row-norm scale ratio in amica-python vs fortran (median=0.9067; expected 1.0)

## Next step
Use the suspect ranking to design a one-line fix in `scripts/parity/adapters/fortran_adapter.py` (most likely the pcakeep slice or a Fortran scaling convention) and re-run the L1 diagnostic in a follow-up cycle. The fix itself is out of scope for this cycle.