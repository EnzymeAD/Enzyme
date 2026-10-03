! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t3 && %t3 | FileCheck %s %}

module custom_forward_derivative_example
  implicit none
  private
  public :: wrapper

  type :: derivative_registration
    procedure(double_value), pointer, nopass :: primal => double_value
    procedure(forward_double_value), pointer, nopass :: forward => forward_double_value
  end type derivative_registration

  type(derivative_registration) :: f__enzyme_register_derivative_double_value

contains

  subroutine double_value(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x
  end subroutine double_value

  subroutine forward_double_value(x, dx, y, dy)
    real, intent(in) :: x, dx
    real, intent(out) :: y, dy

    call double_value(x, y)
    ! Use the derivative of log1p(x), which is 1/(1+x).
    dy = dx / (1.0 + x)
  end subroutine forward_double_value

  subroutine wrapper(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    call double_value(x, y)
  end subroutine wrapper

end module custom_forward_derivative_example

program main
  use enzyme, only: enzyme_dup, enzyme_fwddiff
  use custom_forward_derivative_example, only: wrapper
  implicit none
  real :: x, dx, y, dy

  x = 2.0
  dx = 1.0
  y = 0.0
  dy = 0.0
  call enzyme_fwddiff(wrapper, enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*, "(a,f6.4)") "y = ", y
  write(*, "(a,f6.4)") "dy = ", dy
end program main

! CHECK: y = 4.0000
! CHECK-NEXT: dy = 0.3333
