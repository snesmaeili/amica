# Patched amica17 Fortran reference

This directory ships the **patched AMICA 1.7** Fortran source we use as the
numerical reference for amica-python. It is **AMICA 1.7 only** — do not mix
in 1.5 sources. Two upstream bugs are fixed:

1. `amica17.f90:1465` (non-MKL path) used `(rho-0.0)` instead of `(rho-1.0)`.
   We always compile with `-DMKL` and provide MKL stubs (`mkl_stubs.f90`)
   so we get the correct branch on machines without Intel MKL.
2. The OMP reduction guard at `amica17.f90:1639-1658` accessed Newton/rho
   thread-local arrays unconditionally even when those features are off.
   Each block is now wrapped in the appropriate `if` guard.

Upstream source: <https://github.com/sccn/amica>. Upstream license:
BSD 2-Clause (see `LICENSE.upstream`).

---

## Running on a Windows laptop (no Compute Canada needed)

The cluster binary (`/home/sesma/refs/sccn-amica/amica17_narval`) is
dynamically linked against `/cvmfs/soft.computecanada.ca/...` libraries
that do not exist outside Narval — it will **not** run on a laptop. You
need to either build inside a Linux environment or use the Docker image.
Pick one of the two paths below.

### Path A — Docker Desktop (easiest, no build tools needed)

1. Install Docker Desktop for Windows (one-time): <https://docs.docker.com/desktop/install/windows-install/>.
   Make sure WSL2 is enabled during setup.
2. From a PowerShell/Command-Prompt window in this directory:

   ```powershell
   docker build -t amica17 .
   docker run --rm -v ${PWD}:/work -w /work amica17 amica17 myparam.param
   ```

   The build takes ~3 min the first time, then is cached. You don't need
   gfortran or MPI installed on Windows itself — they live inside the
   container.

### Path B — WSL2 Ubuntu (smaller download than Docker, native binary)

1. Install WSL2 + Ubuntu (one-time, in PowerShell as admin):

   ```powershell
   wsl --install -d Ubuntu
   ```

   Reboot, finish the Ubuntu setup.
2. Open the Ubuntu terminal and install the build tools:

   ```bash
   sudo apt-get update
   sudo apt-get install -y gfortran libopenmpi-dev libopenblas-dev
   ```
3. From inside WSL, navigate to this directory (Windows files are at
   `/mnt/c/...`) and build:

   ```bash
   cd /mnt/c/path/to/amica-benchmark/fortran
   BLAS=openblas ./build.sh
   ```

   This produces `./amica17` — a native Linux binary that runs inside WSL.
   Pass it a parameter file the same way:

   ```bash
   ./amica17 /path/to/myparam.param
   ```

   **Note**: AMICA writes large output files. Keep the run inside WSL's
   own filesystem (`~/work/...` not `/mnt/c/...`) for ~5–10× faster I/O.

---

## File manifest

| File | Purpose |
|---|---|
| `amica17_patched.f90` | The patched amica17 source (155 KB) |
| `funmod2.f90` | Helper module (`gamln`, `psifun`, `matout`) |
| `mkl_stubs.f90` | `vdLn`/`vdExp`/`vdAbs` stubs that route to standard Fortran intrinsics, used because we always compile with `-DMKL` |
| `build.sh` | Two-toolchain build script (FlexiBLAS or OpenBLAS) |
| `Dockerfile` | Multi-stage Ubuntu 22.04 image; final image has `amica17` on PATH |
| `LICENSE.upstream` | BSD-2-Clause from sccn/amica |

## Verifying the build

After building, use any captured fixture to check the binary works.
Example (from `amica-python/tests/fixtures/`):

```bash
# Use synthetic_nondegenerate.param + synthetic_nondegenerate.fdt as a
# 5-source non-degenerate test case. Pre-captured Fortran output is in
# fortran_nondegenerate/ for direct comparison.

./amica17 /path/to/synthetic_nondegenerate.param
diff -r out/ /path/to/fortran_nondegenerate/   # numerical agreement
```

The full sub-01 (118 ch × 1.2 M samples) parity test, including the param
file and pre-processed `.fdt`, lives at
`amica-benchmark/results/post_f1_audit/`.
