! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme -passes="preserve-nvvm" -S %t.bc | FileCheck %s --check-prefix=IR --implicit-check-not=__enzyme_register_derivative
! RUN: %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" %t.bc -o %t.ll
! RUN: %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 %t.ll -o %t2 && %t2 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t3 && %t3 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t4 && %t4 | FileCheck %s %}

module enzyme_test_forward_custom_rule_procedure_pointer
  implicit none
  private
  public :: wrapper, forward_calls

  integer :: forward_calls = 0

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

    forward_calls = forward_calls + 1
    call double_value(x, y)
    ! Use the derivative of log1p(x) to identify the custom rule.
    dy = dx / (1.0 + x)
  end subroutine forward_double_value

  subroutine wrapper(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    call double_value(x, y)
  end subroutine wrapper

end module enzyme_test_forward_custom_rule_procedure_pointer

program main
  use enzyme, only: enzyme_dup, enzyme_fwddiff
  use enzyme_test_forward_custom_rule_procedure_pointer, only: wrapper, forward_calls
  implicit none
  real :: x, dx, y, dy

  x = 2.0
  dx = 3.0
  y = 0.0
  ! Check that the custom rule replaces the previous output derivative.
  dy = 7.0
  call enzyme_fwddiff(wrapper, enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*, "(a,f6.4,a,f6.4,a,i0)") &
    "scaled y = ", y, ", dy = ", dy, ", calls = ", forward_calls
  write(*, "(a,f6.4)") "input dx = ", dx

  forward_calls = 0
  dx = 0.0
  dy = 7.0
  call enzyme_fwddiff(wrapper, enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*, "(a,f6.4,a,f6.4,a,i0)") &
    "zero y = ", y, ", dy = ", dy, ", calls = ", forward_calls
end program main

! IR: !enzyme_derivative
! CHECK: scaled y = 4.0000, dy = 1.0000, calls = 1
! CHECK-NEXT: input dx = 3.0000
! CHECK-NEXT: zero y = 4.0000, dy = 0.0000, calls = 1
