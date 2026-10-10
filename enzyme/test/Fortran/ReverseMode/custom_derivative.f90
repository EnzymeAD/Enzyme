! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t3 && %t3 | FileCheck %s %}

module custom_derivative_example
  implicit none
  private
  public :: wrapper

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

    ! Preserve dy because Enzyme can supply its seed before this call.
    call double_value(x, y)
  end subroutine augment_double_value

  subroutine reverse_double_value(x, dx, y, dy)
    real, intent(in) :: x, y
    real, intent(inout) :: dx, dy

    ! Use the derivative of log1p(x), which is 1/(1+x).
    dx = dx + dy / (1.0 + x)
    ! Consume the derivative of the overwritten output.
    dy = 0.0
  end subroutine reverse_double_value

  function wrapper(x) result(y)
    real, intent(in) :: x
    real :: y
    call double_value(x, y)
  end function wrapper

end module custom_derivative_example

program main
  use enzyme, only: enzyme_autodiff
  use custom_derivative_example, only: wrapper
  implicit none
  real :: x, dx

  x = 2.0
  write(*, "(a,f6.4)") "y = ", wrapper(x)
  dx = 0.0
  call enzyme_autodiff(wrapper, x, dx)
  write(*, "(a,f6.4)") "dx = ", dx
end program main

! CHECK: y = 4.0000
! CHECK-NEXT: dx = 0.3333
