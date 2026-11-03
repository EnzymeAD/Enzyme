! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! ETIME overwrites its arguments VALUES and TIME. The values that they held
! before the call get no adjoint from the uses after the call.

program main
  use enzyme, only: enzyme_autodiff
  implicit none
  real :: x, dx

  x = 3
  dx = 0
  call enzyme_autodiff(f, x, dx)
  write(*,"(f6.2)") dx

contains

  real function f(x)
    real, intent(in) :: x
    real :: v(2), t, s

    v(1) = x
    v(2) = x
    t = x
    s = v(1) * v(2)
    call etime(v, t)
    ! df/dx is 2 * x: v and t no longer depend on x.
    f = s + v(1) + v(2) + t
  end function f

end program main

! CHECK: 6.00
