! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme -passes="preserve-nvvm" -S %t.bc | FileCheck %s --check-prefix=IR --implicit-check-not=__enzyme_register_gradient
! RUN: %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" %t.bc -o %t.ll
! RUN: %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 %t.ll -o %t2 && %t2 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t3 && %t3 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t4 && %t4 | FileCheck %s %}

module enzyme_test_reverse_custom_rule_procedure_pointer
  implicit none
  private
  public :: single_call, repeated_output, augment_calls, reverse_calls

  integer :: augment_calls = 0, reverse_calls = 0

  type :: gradient_registration
    procedure(double_value), pointer, nopass :: primal => double_value
    procedure(augment_double_value), pointer, nopass :: augmented => augment_double_value
    procedure(reverse_double_value), pointer, nopass :: reverse => reverse_double_value
  end type gradient_registration

  type(gradient_registration) :: f__enzyme_register_gradient_double_value

contains

  subroutine double_value(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x
  end subroutine double_value

  subroutine augment_double_value(x, dx, y, dy)
    real, intent(in) :: x, dx
    real, intent(out) :: y
    real, intent(inout) :: dy

    ! Preserve the output derivative for the reverse routine.
    augment_calls = augment_calls + 1
    y = 2.0 * x
  end subroutine augment_double_value

  subroutine reverse_double_value(x, dx, y, dy)
    real, intent(in) :: x, y
    real, intent(inout) :: dx, dy

    ! Use the derivative of log1p(x) to identify the custom rule.
    reverse_calls = reverse_calls + 1
    dx = dx + dy / (1.0 + x)
    dy = 0.0
  end subroutine reverse_double_value

  function single_call(x) result(y)
    real, intent(in) :: x
    real :: y
    call double_value(x, y)
  end function single_call

  function repeated_output(x) result(y)
    real, intent(in) :: x
    real :: y
    call double_value(x, y)
    call double_value(x, y)
  end function repeated_output

end module enzyme_test_reverse_custom_rule_procedure_pointer

program main
  use enzyme, only: enzyme_autodiff
  use enzyme_test_reverse_custom_rule_procedure_pointer, only: &
    single_call, repeated_output, augment_calls, reverse_calls
  implicit none
  real :: x, dx

  x = 2.0
  dx = 0.0
  call enzyme_autodiff(single_call, x, dx)
  write(*, "(a,f6.4,2(a,i0))") &
    "single dx = ", dx, ", augment = ", augment_calls, ", reverse = ", reverse_calls

  dx = 2.0
  call enzyme_autodiff(repeated_output, x, dx)
  write(*, "(a,f6.4)") "overwrite dx = ", dx
end program main

! IR-DAG: !enzyme_augment
! IR-DAG: !enzyme_gradient

! CHECK: single dx = 0.3333, augment = 1, reverse = 1
! CHECK-NEXT: overwrite dx = 2.3333
