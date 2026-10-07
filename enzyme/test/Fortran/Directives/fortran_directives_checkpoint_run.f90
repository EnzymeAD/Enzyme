! REQUIRES: flang_directives
! RUN: %if flangenzyme %{ %fc -cpp %flangDirectives -O0 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t0 && %t0 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -cpp %flangDirectives -O2 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t2 && %t2 | FileCheck %s %}

! A time loop over module state, checkpointed with !$enzyme checkpoint and
! each built-in schedule the plugin names: the gradient must be that of the
! plain loop, differentiated through every step, and the state after it the
! same.

module tl
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  public
  integer, parameter :: m = 5
  real(real64) :: u(m) = 0, p(m) = 0
contains
  subroutine step()
    real(real64) :: tmp(m)
    integer :: k
    do k = 1, m
      tmp(k) = 0.6d0 * sin(u(k) + 0.5d0 * u(mod(k, m) + 1)) + p(k)**2
    end do
    u = tmp
  end subroutine step

  subroutine init(x)
    real(real64), intent(in) :: x(m)
    u = 0.1d0
    p = x
  end subroutine init

  real(real64) function loss()
    loss = sum(u**3)
  end function loss
end module tl

! allow(procedure-not-in-module)
subroutine plain(x, y, n)
  use, intrinsic :: iso_fortran_env, only: real64
  use tl, only: m, init, step, loss
  implicit none
  real(real64), intent(in) :: x(m)
  real(real64), intent(out) :: y
  integer, intent(in) :: n
  integer :: t
  call init(x)
  do t = 1, n
    call step()
  end do
  y = loss()
end subroutine plain

! allow(procedure-not-in-module)
subroutine revolve(x, y, n)
  use, intrinsic :: iso_fortran_env, only: real64
  use tl, only: m, init, step, loss
  implicit none
  real(real64), intent(in) :: x(m)
  real(real64), intent(out) :: y
  integer, intent(in) :: n
  integer :: t
  call init(x)
  !$enzyme checkpoint schedule(revolve) budget(3)
  do t = 1, n
    call step()
  end do
  y = loss()
end subroutine revolve

! allow(procedure-not-in-module)
subroutine periodic(x, y, n)
  use, intrinsic :: iso_fortran_env, only: real64
  use tl, only: m, init, step, loss
  implicit none
  real(real64), intent(in) :: x(m)
  real(real64), intent(out) :: y
  integer, intent(in) :: n
  integer :: t
  call init(x)
  !$enzyme checkpoint schedule(periodic) budget(4)
  do t = 1, n
    call step()
  end do
  y = loss()
end subroutine periodic

program main
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_dup, enzyme_const
  use tl, only: m, u
  implicit none
  real(real64) :: x(m), dx(m), want(m), y, dy, ustate(m)
  external :: plain, revolve, periodic
  integer :: k, n
  logical :: ok

  ok = .true.
  do n = 1, 20, 6
    do k = 1, m
      x(k) = 0.2d0 + 0.1d0 * k
    end do
    want = 0
    dy = 1
    call enzyme_autodiff(plain, enzyme_dup, x, want, enzyme_dup, y, dy, enzyme_const, n)
    ustate = u
    dx = 0
    dy = 1
    call enzyme_autodiff(revolve, enzyme_dup, x, dx, enzyme_dup, y, dy, enzyme_const, n)
    ok = ok .and. maxval(abs(dx - want)) <= 1d-13 * maxval(abs(want)) .and. all(u == ustate)
    dx = 0
    dy = 1
    call enzyme_autodiff(periodic, enzyme_dup, x, dx, enzyme_dup, y, dy, enzyme_const, n)
    ok = ok .and. maxval(abs(dx - want)) <= 1d-13 * maxval(abs(want)) .and. all(u == ustate)
  end do
  if (ok) then
    print '(a)', 'checkpointed gradients ok'
  else
    print '(a)', 'checkpointed gradients differ'
  end if
end program main

! CHECK: checkpointed gradients ok
