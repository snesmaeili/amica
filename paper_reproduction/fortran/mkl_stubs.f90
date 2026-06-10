! Stub implementations of MKL vector math functions using standard Fortran.
! These replace Intel MKL's vdLn, vdExp etc. for compilation with gfortran.

subroutine vdLn(n, x, y)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: y(n)
  y = log(x)
end subroutine vdLn

subroutine vdExp(n, x, y)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: y(n)
  y = exp(x)
end subroutine vdExp

subroutine vdAbs(n, x, y)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: y(n)
  y = abs(x)
end subroutine vdAbs
