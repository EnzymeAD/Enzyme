! !DIR$ ENZYME custom_rule across translation units: the rule procedures are
! PRIVATE module procedures, which the module file still carries (with the
! directive), so a unit that only USEs the module registers the rule too.
! There the rules are just declarations; they must outlive the optimizations
! that run before Enzyme. double_value is 2x but differentiates like log1p:
! 1/(1+x) = 0.3333 at x = 2.
!
! REQUIRES: flang_enzyme_mlir, flangenzyme
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_enzyme_driver -cpp -DRULES -O0 -mmlir -enzyme-flang-mlir-ad=false -module-dir %t -c %s -o %t/rules.o
! RUN: FileCheck %s --check-prefix=MOD < %t/private_rules.mod
! RUN: %flang_enzyme_driver -cpp -O0 -mmlir -enzyme-flang-mlir-ad=false %loadFlangEnzyme %loadFortran -I%t %s %t/rules.o -o %t/a0 && %t/a0 | FileCheck %s
! RUN: %flang_enzyme_driver -cpp -DRULES -O2 -mmlir -enzyme-flang-mlir-ad=false -module-dir %t -c %s -o %t/rules2.o
! RUN: %flang_enzyme_driver -cpp -O2 -mmlir -enzyme-flang-mlir-ad=false %loadFlangEnzyme %loadFortran -I%t %s %t/rules2.o -o %t/a2 && %t/a2 | FileCheck %s

! MOD: private::augment_double_value
! MOD: private::reverse_double_value
! MOD: !dir$ enzyme custom_rule(double_value, augmented=augment_double_value, reverse=reverse_double_value)
! MOD: subroutine augment_double_value(x,dx,y,dy)
! MOD: subroutine reverse_double_value(x,dx,y,dy)

#ifdef RULES
module private_rules
  implicit none
  private
  public :: double_value
  !dir$ enzyme custom_rule(double_value, augmented=augment_double_value, reverse=reverse_double_value)
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
end module
#else
module user
  use private_rules, only: double_value
  implicit none
contains
  real function wrapper(x)
    real, intent(in) :: x
    call double_value(x, wrapper)
  end function
end module

program main
  use enzyme, only: enzyme_autodiff
  use user
  implicit none
  real :: x, dx
  x = 2.0
  dx = 0.0
  call enzyme_autodiff(wrapper, x, dx)
  print '(F6.4)', dx
end program
#endif

! CHECK: 0.3333
