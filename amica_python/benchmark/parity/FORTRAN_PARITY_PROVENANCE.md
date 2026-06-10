# AMICA 1.7 Fortran reference — build & parity provenance

Validated Fortran **AMICA 1.7** reference binary for amica-python numerical parity.
Built on Compute Canada **fir**, 2026-06-09.

Durable copy: `/project/rrg-kjerbi/sesma/amica_fortran_reference/`
(binary + `src/` + `build.sh` + `submit_parity.sbatch` + `parity_result_synth6_50k.json`
+ `SHA256SUMS`). `/scratch` is wiped after 60 days — use the `/project` copy.

## What this is
`amica17` = AMICA 1.7 (Palmer / SCCN, <https://github.com/sccn/amica>), the version
amica-python reimplements. Source = `amica-python-benchmark/fortran/amica17_patched.f90`
plus the three fixes below. The binary is dynamically linked against fir's cvmfs gentoo
libraries — it runs on **fir compute nodes**, not on a laptop.

Binary sha256: `c02f22c37cb259364e921d1e1b42f7181ce9fb7baae6a716c2ade261b49771fe`
(full list in `SHA256SUMS`).

## Rebuild (one command)
```bash
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
cd src && bash build.sh          # -> ./amica17
```
Flags: `-cpp -DMKL -O2 -fopenmp -ffree-line-length-none -fallow-argument-mismatch`;
link `funmod2.o mkl_stubs.o amica17_p.o -lflexiblas`.

## Three fixes vs raw upstream amica17.f90 (all required for fix_init parity)
1. **`-DMKL` + `mkl_stubs.f90`** — the non-MKL (`#else`) path at `amica17.f90:1465`
   computes the generalized-Gaussian **score** with exponent `(rho-0.0)` instead of
   `(rho-1.0)`. The score is used only in the *gradient/update*, never in the
   likelihood — so iter-0 LL matched to 1e-15 while every update stepped the wrong way
   (LL fell each iter → lrate halved to underflow → "stall" at iter 162 with a *growing*
   gradient norm). Compiling with `-DMKL` selects the correct `(rho-1.0)` branch;
   `mkl_stubs.f90` supplies `vdLn`/`vdExp`/`vdAbs` via libm. **This was the root cause.**
2. **OMP reduction guards** (`amica17_patched.f90`) — upstream reduces Newton/rho
   thread-local arrays unconditionally; the patched source wraps each in its `if`
   (only matters when `do_newton=.false.`).
3. **`comp_list` init under `fix_init`** (`amica17_patched.f90`, fix_init branch ~L813) —
   upstream sets `comp_list` only in the random-init and `load_A` branches, so
   `get_unmixing_matrices` reads uninitialized `comp_list` under `fix_init=1` → segfault.
   Added `comp_list(i,h) = (h-1)*nw + i` to the fix_init loop.

## Fixed-init parity result (synth6, 6ch × 50k, m=3, Newton; shared sphere+mean+fix_init)
Identical start state isolates pure algorithmic agreement. From `parity_result_synth6_50k.json`:

| metric | value |
|---|---|
| iter-0 LL abs diff | 3.6e-15 |
| final LL abs diff | 6.97e-08  (rel 3.4e-06 %) |
| W matched \|r\| (mean / min) | 0.99999999992 / 0.99999999981 |
| matched-source \|r\| | 0.99999999992 |
| W aligned Frobenius rel | 1.2e-05 |
| Fortran n_iter | 1000 (converged, grad-norm → 1e-5) |

**Conclusion:** amica-python reproduces AMICA 1.7 to optimizer precision.

## Re-running the parity end to end
`submit_parity.sbatch` does: prep (numpy) → build (`-DMKL`) → run `amica17` →
amica-python compare. It imports `amica_python.benchmark.parity.run_fortran_parity`
(harness lives in the amica-python repo). Submit on fir:
```bash
sbatch /project/rrg-kjerbi/sesma/amica_fortran_reference/submit_parity.sbatch
```
(account `def-kjerbi`, CPU, ~2 min. Paths inside the script point at the build/work
dirs under `/scratch/sesma/fortran_parity/`; adjust if relocating.)
