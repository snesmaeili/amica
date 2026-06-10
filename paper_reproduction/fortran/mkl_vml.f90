! Empty MKL VML include file for gfortran builds.
!
! Upstream amica17_patched.f90 line 22 has `include 'mkl_vml.f90'` inside
! the `#ifdef MKL` block to bring in Intel MKL vector-math interface
! declarations. We compile with `-DMKL` (to take the rho-1.0 branch) but
! we DON'T link against Intel MKL — instead, mkl_stubs.f90 provides
! gfortran-native implementations of vdLn/vdExp/vdAbs that the linker
! resolves separately. So this include needs to be a no-op: no
! declarations needed (gfortran allows implicit external linkage for the
! stub subroutines).
