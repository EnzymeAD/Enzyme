! The plugin parses each directive and reports syntax errors. A directive the
! plugin rejects is a parse error, so semantics does not run (see
! fortran_directives_errors.f90 for the errors flang reports on the names).
!
! REQUIRES: flang_directives
! RUN: not %fc -fc1 %flangFc1Directives -fsyntax-only %s 2>&1 | FileCheck %s

module m
  implicit none
  real :: v
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'backward' is not an argument of 'custom_rule'
  !dir$ enzyme custom_rule(f, backward=g)
  ! CHECK: [[@LINE+1]]:{{.*}}error: unknown directive 'frobnicate'
  !dir$ enzyme frobnicate(f)
  ! CHECK: [[@LINE+1]]:{{.*}}error: 'shadow' requires argument 'shadow'
  !dir$ enzyme shadow(v)
  ! CHECK: [[@LINE+1]]:{{.*}}error: custom_rule needs both augmented= and reverse= for reverse mode
  !dir$ enzyme custom_rule(f, augmented=g)
contains
  subroutine f()
  end subroutine
  subroutine g()
  end subroutine
end module
