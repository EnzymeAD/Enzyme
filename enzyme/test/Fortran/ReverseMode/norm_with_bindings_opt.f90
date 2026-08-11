! REQUIRED: fortran
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none
  integer, parameter :: n = 1000000
  integer, parameter :: initial_value = 20
  real :: x(n), dx(n)
  real :: y(n), dy(n), yp

  x(:) = initial_value
  dy(:) = 1.0

  call norm(n, x, y)

  ! Rescale the output to avoid compiler-specific output formatting
  yp = y(n) * 1.0e+06
  write(*,"(f6.4)") yp

  dx(:) = 0.0
  call enzyme_autodiff(norm, enzyme_const, n, &
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(f6.4)") dy(n)

contains

  ! TODO: Switch to assumed shape implementation once
  !       https://github.com/EnzymeAD/Enzyme/issues/2820
  !       has been addressed
  subroutine norm(n, x, y)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    real, dimension(n), intent(out) :: y
    y(:) = x / sum(x)
  end subroutine

end program

! CHECK: 1.0000
! CHECK-NEXT: 0.0000
