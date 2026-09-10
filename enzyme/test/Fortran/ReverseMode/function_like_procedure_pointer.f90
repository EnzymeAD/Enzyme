! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes="preserve-nvvm" -S | FileCheck %s --check-prefix=IR --implicit-check-not=__enzyme_function_like__
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_autodiff
  implicit none

  ! A procedure pointer initializer cannot target an internal procedure.
  ! Use an external function with an explicit interface to avoid a module.
  interface
    function double_value(x) result(y)
      real, value :: x
      real :: y
    end function double_value
  end interface

  procedure(double_value), pointer :: &
    fn__enzyme_function_like__log1p => double_value

  real :: x, dx

  x = 2.0
  dx = 0.0
  call enzyme_autodiff(test, x, dx)

  write(*,"(f6.4)") dx

contains

  function test(x) result(y)
    real, intent(in) :: x
    real :: y

    y = double_value(x)
  end function test

end program main

function double_value(x) result(y)
  implicit none
  real, value :: x
  real :: y

  y = 2.0 * x
end function double_value

! IR: "enzyme_math"="log1p"
! CHECK: 0.3333
