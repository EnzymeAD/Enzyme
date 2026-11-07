! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.bc && not %opt %loadEnzyme %enzyme %t.bc -disable-output 2>&1 | FileCheck %s
! RUN: %if flangenzyme %{ not %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 2>&1 | FileCheck %s %}

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none

  integer, parameter :: n = 4
  real :: x(n), y(n), z
  real :: dx(n), dy(n), dz

  x(:) = 2.0
  y(:) = 3.0
  z = 1.0
  dx(:) = 0.0
  dy(:) = 0.0
  dz = 1.0

  call enzyme_autodiff(dot, enzyme_const, n, &
                       enzyme_dup, x, dx, &
                       enzyme_dup, y, dy, &
                       enzyme_dup, z, dz)
  write(*, "(f4.1)") dx(1)
  write(*, "(f4.1)") dy(1)
  write(*, "(f4.1)") dz

contains

  real function dot(n, a, b, c)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: a
    real, dimension(n), intent(in) :: b
    real, intent(in) :: c
    dot = dot_product(a, b) + c
  end function dot

end program main

! CHECK: error: {{.*}}Enzyme: No reverse pass found for _FortranADotProductReal4
