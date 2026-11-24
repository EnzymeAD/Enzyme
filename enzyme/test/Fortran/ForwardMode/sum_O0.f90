! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Forward mode through the flang runtime's SUM (_FortranASumReal4), which
! flang calls at -O0.

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
  implicit none
  integer :: n
  real :: x(4), dx(4), y, dy

  n = 4
  x = [1, 2, 3, 4]
  dx = [1, 10, 100, 1000]
  dy = 0
  call enzyme_fwddiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                      enzyme_dup, y, dy)
  print *, int(y)
  print *, int(dy)

contains

  subroutine f(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: y
    real, allocatable :: t(:)
    t = 2 * x
    y = sum(x, mask=x > 1.5) + sum(t)
  end subroutine f

end program main

! CHECK: 29
! CHECK-NEXT: 3332
