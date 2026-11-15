! Flang resolves the names that the plugin returns and checks their kinds.
! (Syntax errors are in fortran_directives_parse_errors.f90.)
!
! REQUIRES: flang_directives
! RUN: mkdir -p %t.mod
! RUN: not %fc -fc1 %flangFc1Directives -module-dir %t.mod -fsyntax-only %s 2>&1 | FileCheck %s

module m
  implicit none
  real :: v
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'nosuch' is not declared
  !dir$ enzyme custom_rule(f, forward=nosuch)
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'v' is not a procedure
  !dir$ enzyme custom_rule(f, forward=v)
contains
  subroutine f()
  end subroutine
  subroutine g()
  end subroutine
end module
