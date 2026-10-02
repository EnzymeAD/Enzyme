! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t0 && %t0 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t2 && %t2 | FileCheck %s %}

! A nonlinear solve iterated to a fixed point, with its state and parameters
! in a module, as MITgcm's streamice keeps its velocities in COMMON blocks.
! The adjoint is iterated at the converged state (Tapenade's FP-LOOP), with a
! Tapenade-style control function. The gradient must match centred finite
! differences of the plain loop. The state lives in module variables, which
! need -enzyme-global-activity.

module model
  implicit none
  integer, parameter :: m = 5
  real(8) :: u(m) = 0, p(m) = 0
  integer :: ncontrol = 0
contains
  logical function step(i, tol)
    integer(8), value :: i
    real(8), intent(in) :: tol
    real(8) :: tmp(m), err
    integer :: k
    do k = 1, m
      tmp(k) = 0.3d0 * sin(u(k) + 0.5d0 * u(mod(k, m) + 1)) + p(k)**2
    end do
    err = 0
    do k = 1, m
      err = max(err, abs(tmp(k) - u(k)))
      u(k) = tmp(k)
    end do
    step = err > tol
  end function step

  real(8) function loss()
    integer :: k
    loss = 0
    do k = 1, m
      loss = loss + u(k)**3
    end do
  end function loss

  subroutine init(x)
    real(8), intent(in) :: x(m)
    integer :: k
    do k = 1, m
      u(k) = 0
      p(k) = x(k)
    end do
  end subroutine init

  ! Tapenade's adFixedPoint_notReduced protocol: cumul is -1 on the first
  ! call, then the squared norm of the adjoint update.
  integer function control(cumul, reduction)
    real(8), intent(inout) :: cumul
    real(8), intent(inout) :: reduction
    real(8), save :: ref = -1
    ncontrol = ncontrol + 1
    control = 1
    if (cumul < 0) then
      ref = -1
    else if (ref < 0) then
      ref = cumul
    else if (cumul <= reduction * ref) then
      control = 0
    end if
  end function control
end module model

subroutine plain(x, y, tol)
  use model
  implicit none
  real(8), intent(in) :: x(m), tol
  real(8), intent(out) :: y
  integer(8) :: i
  call init(x)
  i = 0
  do while (step(i, tol))
    i = i + 1
  end do
  y = loss()
end subroutine plain

subroutine fixed(x, y, tol)
  use enzyme
  use model
  implicit none
  real(8), intent(in) :: x(m), tol
  real(8), intent(out) :: y
  call init(x)
  call enzyme_fixed_point(step, enzyme_fp_state, u, int(8 * m, 8), &
                          enzyme_fp_reduction, 1d-24, &
                          enzyme_fp_control, control, tol)
  y = loss()
end subroutine fixed

program main
  use enzyme
  use model
  implicit none
  real(8) :: x(m), dx(m), y, dy, yp, ym, fd(m), h, tol
  external :: fixed
  integer :: k
  logical :: ok

  tol = 1d-15
  do k = 1, m
    x(k) = 0.2d0 + 0.1d0 * k
  end do

  ! Finite differences of the plain loop.
  h = 1d-6
  do k = 1, m
    x(k) = x(k) + h
    call plain(x, yp, tol)
    x(k) = x(k) - 2 * h
    call plain(x, ym, tol)
    x(k) = x(k) + h
    fd(k) = (yp - ym) / (2 * h)
  end do

  do k = 1, m
    dx(k) = 0
  end do
  dy = 1
  call enzyme_autodiff(fixed, enzyme_dup, x, dx, enzyme_dup, y, dy, &
                       enzyme_const, tol)

  ok = all(abs(dx - fd) <= 1d-6 * abs(fd))
  if (ncontrol < 3) ok = .false.
  if (.not. ok) then
    print *, "gradient", dx
    print *, "expected", fd
    print *, "control calls", ncontrol
  end if
  ! CHECK: ok
  if (ok) print "(a)", "ok"
end program main
