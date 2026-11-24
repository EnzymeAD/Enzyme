! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through the flang runtime's array assignment (_FortranAAssign)
! and SUM (_FortranASumReal4), which flang calls at -O0.

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none
  integer :: n
  real :: x(4), dx(4), y, dy

  n = 4
  x = [1, 2, 3, 4]

  dx = 0
  dy = 1
  call enzyme_autodiff(realloc, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)
  print *, int(y)
  print *, int(dx(1)), int(dx(2)), int(dx(3)), int(dx(4))

  dx = 0
  dy = 1
  call enzyme_autodiff(convert, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)
  print *, int(y)
  print *, int(dx(1)), int(dx(2)), int(dx(3)), int(dx(4))

contains

  subroutine realloc(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: y
    real, allocatable :: t(:)
    integer :: k
    ! Allocates t from a strided section.
    t = 2 * x(1:n:2)
    y = sum(t, mask=t > 3)
    ! Reallocates t with another shape in every iteration.
    do k = 1, 2
      t = x(1:k+1) * k
      y = y + sum(t)
    end do
  end subroutine realloc

  subroutine convert(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: y
    real, allocatable :: t(:)
    real(8), allocatable :: r(:)
    ! REAL(8) = REAL(4) into an allocated array.
    allocate(r(n))
    r = x
    ! Broadcast of a scalar.
    allocate(t(n))
    t = x(n)
    t(1:2) = t(1:2) + real(r(1:2) * r(1:2))
    ! Overlapping sections.
    t(1:3) = t(2:4)
    y = sum(t, mask=t > 4.5)
  end subroutine convert

end program main

! CHECK: 21
! CHECK-NEXT: 3 3 4 0
! CHECK-NEXT: 8
! CHECK-NEXT: 0 4 0 1
