! The plugin registers the arguments of each directive; flang checks them.
!
! REQUIRES: flang_directives
! RUN: mkdir -p %t.mod
! RUN: not %fc -fc1 %flangFc1Directives -module-dir %t.mod -fsyntax-only %s 2>&1 | FileCheck %s

module m
  implicit none
  public
  real :: v
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'nosuch' is not declared
  !dir$ enzyme custom_rule(f, forward=nosuch)
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'v' is not a procedure
  !dir$ enzyme custom_rule(f, forward=v)
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'backward' is not an argument of the 'enzyme custom_rule' directive
  !dir$ enzyme custom_rule(f, backward=g)
  ! CHECK: [[@LINE+1]]:{{.*}}error: Unknown 'enzyme' directive 'frobnicate'
  !dir$ enzyme frobnicate(f)
  ! CHECK: [[@LINE+1]]:{{.*}}error: The 'enzyme shadow' directive requires argument 'shadow'
  !dir$ enzyme shadow(v)
contains
  subroutine f()
  end subroutine f
  subroutine g()
  end subroutine g
end module m
