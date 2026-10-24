! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! ETIME overwrites its arguments VALUES and TIME. After the call, their
! tangents are zero.

program main
  use enzyme, only: enzyme_dup, enzyme_fwddiff
  implicit none
  real :: x, dx, y, dy

  x = 3
  dx = 1
  dy = 0
  call enzyme_fwddiff(g, enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(f6.2)") dy

contains

  subroutine g(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    real :: v(2), t

    v(1) = x
    v(2) = x
    t = x
    call etime(v, t)
    ! dy/dx is 2 * x: v and t no longer depend on x.
    y = x * x + v(1) + v(2) + t
  end subroutine g

end program main

! CHECK: 6.00
