! Errors for !$enzyme fixed_point. Where the directive is is checked after
! names are resolved, and only if they are, so the arguments are checked in a
! part of their own. A directive after the last declaration goes with the
! loop that starts the execution part; one between declarations, or in a
! scope without an execution part, has no loop to go with.
!
! REQUIRES: flang_directives
! RUN: not %fc -fc1 %flangFc1Directives -cpp -DPART=1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=WHERE
! RUN: not %fc -fc1 %flangFc1Directives -cpp -DPART=2 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ARGS

#if PART == 1
subroutine where(x, n)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n
  integer :: i
  real(real64), intent(inout) :: x(n)
  real(real64) :: y
  ! WHERE: error: A DO or DO WHILE loop must follow the 'ENZYME FIXED_POINT' directive
  !$enzyme fixed_point(x)
  y = 1
  ! WHERE: error: A DO or DO WHILE loop must follow the 'ENZYME FIXED_POINT' directive
  !$enzyme fixed_point(x)
  do concurrent (i = 1:n)
    x(i) = 0
  end do
  ! Another directive in between is fine.
  !$enzyme fixed_point(x)
  !dir$ unroll
  do i = 1, n
    x(i) = x(i) * y
  end do
  ! WHERE: error: A DO or DO WHILE loop must follow the 'ENZYME FIXED_POINT' directive
  !$enzyme fixed_point(x)
end subroutine where

subroutine declarations(x)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  real(real64), intent(inout) :: x
  ! WHERE: error: A DO or DO WHILE loop must follow the 'ENZYME FIXED_POINT' directive
  !$enzyme fixed_point(x)
  integer :: k
  do k = 1, 2
    x = x * 0.5d0
  end do
end subroutine declarations

module no_loops
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  public
  real(real64) :: q
  ! WHERE: error: A DO or DO WHILE loop must follow the 'ENZYME FIXED_POINT' directive
  !$enzyme fixed_point(q)
end module no_loops
#endif

#if PART == 2
subroutine args(x, n)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n
  integer :: i
  real(real64), intent(inout) :: x(n)
  real(real64) :: y
  ! ARGS: error: A 'enzyme fixed_point' directive needs at least 1 variable(s)
  !$enzyme fixed_point max_iters(3)
  do i = 1, n
  end do
  ! ARGS: error: 'y' is not a procedure
  !$enzyme fixed_point(x) control(y)
  do i = 1, n
  end do
  ! ARGS: error: Argument 'reduction' must be a real or integer literal
  !$enzyme fixed_point(x) reduction('small')
  do i = 1, n
  end do
  ! ARGS: error: Argument 'max_iters' must be an integer
  !$enzyme fixed_point(x) max_iters(1.5)
  do i = 1, n
  end do
  ! ARGS: error: The variables of a 'enzyme fixed_point' directive must come before its other arguments
  !$enzyme fixed_point(max_iters=3, x)
  do i = 1, n
  end do
  ! ARGS: error: 'args' is not a variable
  !$enzyme fixed_point(args)
  do i = 1, n
  end do
  ! ARGS: error: 'z' is not declared
  !$enzyme fixed_point(z)
  do i = 1, n
  end do
  ! ARGS: error: 'tolerance' is not an argument of the 'enzyme fixed_point' directive
  !$enzyme fixed_point(x) tolerance(1d-8)
  do i = 1, n
  end do
end subroutine args
#endif
