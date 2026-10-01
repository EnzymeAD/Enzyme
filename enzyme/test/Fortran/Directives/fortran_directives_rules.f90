! !DIR$ ENZYME custom_rule, inactive and shadow, end to end.
! - custom_rule replaces the registration variable of the Fortran bindings:
!   double_value is 2x but differentiates like log1p, 1/(1+x) = 0.3333 at 2.
!   LLVM route: this plugin only turns the directives into markers.
! - inactive on scale: d/dx (x * scale(x)) with scale(x) = x taken as
!   constant is scale(3) = 3, in both routes.
! - shadow pairs a module variable with the shadow the program seeds: the
!   forward derivative of x * g along g_d = 1 is x = 3.
!
! REQUIRES: flang_directives, flangenzyme
! RUN: %fc %flangDirectives -O0 %loadFlangEnzyme -mllvm -enzyme-global-activity %loadFortran %s -o %t0 && %t0 | FileCheck %s --check-prefixes=CHECK,LLVM
! RUN: %fc %flangDirectives -O2 %loadFlangEnzyme -mllvm -enzyme-global-activity %loadFortran %s -o %t2 && %t2 | FileCheck %s --check-prefixes=CHECK,LLVM

module rules
  implicit none
  real :: g = 2.0, g_d = 0.0
  !dir$ enzyme custom_rule(double_value, augmented=augment_double_value, reverse=reverse_double_value)
  !dir$ enzyme shadow(g, shadow=g_d)
contains
  subroutine double_value(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x
  end subroutine
  subroutine augment_double_value(x, dx, y, dy)
    real, intent(in) :: x, dx
    real, intent(out) :: y
    real, intent(inout) :: dy
    call double_value(x, y)
  end subroutine
  subroutine reverse_double_value(x, dx, y, dy)
    real, intent(in) :: x, y
    real, intent(inout) :: dx, dy
    dx = dx + dy / (1.0 + x)
    dy = 0.0
  end subroutine
  real function wrapper(x)
    real, intent(in) :: x
    call double_value(x, wrapper)
  end function

  real function scale(x)
    real, intent(in) :: x
    !dir$ enzyme inactive
    scale = x
  end function
  real function scaled(x)
    real, intent(in) :: x
    scaled = x * scale(x)
  end function

  real function times_g(x)
    real, intent(in) :: x
    times_g = x * g
  end function
end module

program main
  use enzyme, only: enzyme_autodiff
  use rules
  implicit none
  real :: x, dx
  real, external :: f__enzyme_fwddiff
  x = 2.0
  dx = 0.0
  call enzyme_autodiff(wrapper, x, dx)
  print '(F6.4)', dx
  x = 3.0
  dx = 0.0
  call enzyme_autodiff(scaled, x, dx)
  print '(F6.4)', dx
  dx = 0.0
  g_d = 1.0
  print '(F6.4)', f__enzyme_fwddiff(times_g, x, dx)
end program

! LLVM: 0.3333
! CHECK: 3.0000
! LLVM: 3.0000
