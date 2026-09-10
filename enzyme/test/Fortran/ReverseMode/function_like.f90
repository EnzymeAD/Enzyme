! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_autodiff, enzyme_function_like, enzyme_log1p
  implicit none

  real :: x, dx

  x = 2.0
  dx = 0.0
  call enzyme_function_like(double_value, enzyme_log1p)
  call enzyme_autodiff(test, x, dx)

  write(*,"(f6.4)") dx

contains

  function double_value(x) result(y)
    real, value :: x
    real :: y

    y = 2.0 * x
  end function double_value

  function test(x) result(y)
    real, intent(in) :: x
    real :: y

    y = double_value(x)
  end function test

end program main

! CHECK: 0.3333
