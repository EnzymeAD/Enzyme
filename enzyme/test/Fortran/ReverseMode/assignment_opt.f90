! REQUIRES: fortran
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! At -O1 and above, flang inlines the array assignments. Each element x(i)
! of a 1-based array is then addressed as a scaled GEP of i followed by a
! constant GEP of minus one element, through which type analysis must find
! the element types of the arrays to type the copies between them.

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none
  integer :: n
  real :: x(4), dx(4), y, dy

  n = 4
  x = [1, 2, 3, 4]
  dx = 0
  y = 0
  dy = 1

  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)

  print *, int(y)
  print *, int(dx(1)), int(dx(2)), int(dx(3)), int(dx(4))

contains

  subroutine f(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: y
    real, allocatable :: t(:)
    real(8), allocatable :: r(:)
    ! Section assignments do not reallocate, so flang inlines them.
    allocate(r(n), t(n))
    r(:) = x
    t(:) = x(n)
    t(:) = t + real(r * r)
    y = sum(t, mask=x > 1.5)
  end subroutine f

end program main

! CHECK: 41
! CHECK-NEXT: 0 4 6 11
