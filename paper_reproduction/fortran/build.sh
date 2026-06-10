#!/usr/bin/env bash
# Build the patched amica17 Fortran reference binary.
#
# The patched source fixes two upstream amica17 bugs:
#   1. amica17.f90:1465 (non-MKL path) uses (rho-0.0) instead of (rho-1.0).
#      We always compile with -DMKL and provide MKL stubs for vdLn/vdExp.
#   2. The OMP reduction guard at amica17.f90:1639-1658 unconditionally
#      accesses thread-local arrays that are only allocated when do_newton
#      or dorho. The patched version wraps each block in the right `if`.
#
# Tested toolchains:
#   - Compute Canada Narval (gcc/12.3 + openmpi/4.1.5 + flexiblas/3.3.1)
#   - Ubuntu 22.04 (gfortran-11 + libopenmpi-dev + libopenblas-dev)
#   - Docker via ./Dockerfile
#
# Usage:
#   ./build.sh            -> produces amica17 in this directory
#   BLAS=openblas ./build.sh
#
# Requirements (Ubuntu/Debian):
#   sudo apt-get install gfortran libopenmpi-dev libopenblas-dev
#
# Requirements (macOS via Homebrew):
#   brew install gcc open-mpi openblas
#   (use gfortran from the gcc formula)

set -euo pipefail

cd "$(dirname "$0")"

BLAS="${BLAS:-flexiblas}"
case "$BLAS" in
  flexiblas) BLAS_LIB="-lflexiblas" ;;
  openblas)  BLAS_LIB="-lopenblas"  ;;
  *)         echo "Unknown BLAS=$BLAS (use flexiblas or openblas)" >&2; exit 1 ;;
esac

FC="${FC:-mpifort}"
FFLAGS="-cpp -DMKL -O2 -fopenmp -ffree-line-length-none -fallow-argument-mismatch"

echo "[build] Compiling funmod2"
$FC $FFLAGS -c funmod2.f90 -o funmod2.o

echo "[build] Compiling MKL stubs"
$FC $FFLAGS -c mkl_stubs.f90 -o mkl_stubs.o

echo "[build] Compiling amica17 (patched)"
$FC $FFLAGS -I. -c amica17_patched.f90 -o amica17_p.o

echo "[build] Linking with $BLAS"
$FC -O2 -fopenmp funmod2.o mkl_stubs.o amica17_p.o $BLAS_LIB -o amica17

echo "[build] OK -> $(pwd)/amica17"
