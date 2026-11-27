! REQUIRES: fortran
! Note: -S -emit-llvm rather than the usual -flto -c, so as not to require an
!       LTO-capable system linker.
! RUN: %fc -O1 %loadFortran -S -emit-llvm %s -o %t.ll && %opt %t.ll %loadEnzyme %enzyme -S -o %t2.ll && %fc -O1 %t2.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -O2 %loadFortran -S -emit-llvm %s -o %t.ll && %opt %t.ll %loadEnzyme %enzyme -S -o %t2.ll && %fc -O2 %t2.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Differentiation through a type-bound procedure (vtable) dispatch.
!
! `dispatch` takes a polymorphic dummy argument, so flang lowers `s%step` to a
! load out of the binding table @_QMsolver_modEXvXsolver_t followed by an
! indirect call. `step_impl` has no direct call site anywhere in the module --
! only a `ptrtoint` entry in that table -- so Enzyme has to recover both it and
! its argument types from the dispatch.

module solver_mod
  implicit none

  type :: solver_t
     real :: scale
   contains
     procedure :: step => step_impl
  end type solver_t

contains

  subroutine step_impl(self, n, x, y)
    class(solver_t), intent(in) :: self
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(inout) :: y
    integer :: i
    y = 0
    do i = 1, n
       y = y + self%scale * x(i) * x(i)
    end do
  end subroutine step_impl

  ! Polymorphic dummy => dynamic dispatch through the binding table.
  subroutine dispatch(s, n, x, y)
    class(solver_t), intent(in) :: s
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(inout) :: y
    call s%step(n, x, y)
  end subroutine dispatch

  subroutine run(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(inout) :: y
    type(solver_t) :: s
    s%scale = 2.0
    call dispatch(s, n, x, y)
  end subroutine run

end module solver_mod

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
  use solver_mod
  implicit none

  integer :: n
  real :: x(3), dx(3), y, dy

  n = 3
  x = [2.0, 3.0, 4.0]
  dx = [1.0, 0.0, 0.0]
  y = 0
  dy = 0

  call enzyme_fwddiff(run, enzyme_const, n, &
                      enzyme_dup, x, dx, enzyme_dup, y, dy)

  ! y  = 2*(2^2 + 3^2 + 4^2) = 58
  ! dy = 2*2*x(1)*dx(1)      = 8
  print *, int(y)
  print *, int(dy)
end program main

! CHECK: 58
! CHECK-NEXT: 8
