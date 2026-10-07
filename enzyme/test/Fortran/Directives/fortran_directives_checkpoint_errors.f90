! The errors of !$enzyme checkpoint.
!
! REQUIRES: flang_directives
! RUN: not %fc -fc1 %flangFc1Directives -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s

module ckerr
  implicit none
  real(8) :: u(5)
contains
  subroutine bad(n)
    integer, intent(in) :: n
    integer :: t
    !$enzyme checkpoint schedule(fastest)
    do t = 1, n
      u = sin(u)
    end do
  end subroutine bad
end module ckerr

! CHECK: enzyme checkpoint: unknown schedule "fastest", expected binomial, revolve, periodic or store_all
