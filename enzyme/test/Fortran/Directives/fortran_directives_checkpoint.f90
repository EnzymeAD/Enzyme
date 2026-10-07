! !$enzyme checkpoint in front of a DO loop: the plugin replaces the marker
! flang put at the start of the loop body with
!   __enzyme_set_checkpointing(schedule, budget)
! the schedule a tag of enzyme/checkpoint_schedule.h (revolve 2, periodic 1,
! the default) and the budget all ones without one, and gives each variable
! of the directive, the data of an allocatable here, to Enzyme as a region
! of the loop's snapshots: __enzyme_ptr_size_hint(data, bytes) in front of
! the loop.
!
! REQUIRES: flang_directives
! RUN: %fc -fc1 %flangFc1Directives -O0 -emit-llvm %s -o - | FileCheck %s

module ck
  implicit none
  real(8) :: u(5)
  real(8), allocatable :: v(:)
contains
  subroutine run(n)
    integer, intent(in) :: n
    integer :: t
    !$enzyme checkpoint(v) schedule(revolve) budget(4)
    do t = 1, n
      u = sin(u) + v(1:5)
      v = 0.5d0 * v
    end do
  end subroutine run

  subroutine dflt(n)
    integer, intent(in) :: n
    integer :: t
    !$enzyme checkpoint
    do t = 1, n
      u = sin(u)
    end do
  end subroutine dflt

  subroutine binom(n)
    integer, intent(in) :: n
    integer :: t
    !dir$ enzyme checkpoint schedule(binomial) budget(3)
    do t = 1, n
      u = cos(u)
    end do
  end subroutine binom
end module ck

! CHECK-LABEL: define void @_QMckPrun(
! CHECK: call void @__enzyme_ptr_size_hint(ptr %{{.+}}, i64 %{{.+}})
! CHECK: call void @__enzyme_set_checkpointing(i64 2, i64 4)
! CHECK-LABEL: define void @_QMckPdflt(
! CHECK-NOT: __enzyme_ptr_size_hint
! CHECK: call void @__enzyme_set_checkpointing(i64 1, i64 -1)
! CHECK-LABEL: define void @_QMckPbinom(
! CHECK: call void @__enzyme_set_checkpointing(i64 4, i64 3)
