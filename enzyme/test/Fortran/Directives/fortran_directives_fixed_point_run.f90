! REQUIRES: flang_directives
! RUN: %if flangenzyme %{ %fc -cpp %flangDirectives -O0 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t0 && %t0 > %t0.out && FileCheck %s < %t0.out %}
! RUN: %if flangenzyme %{ %fc -cpp -DROUTE1 %flangDirectives -O0 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t0r && %t0r > %t0r.out && diff %t0.out %t0r.out %}
! RUN: %if flangenzyme %{ %fc -cpp %flangDirectives -O2 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t2 && %t2 > %t2.out && FileCheck %s < %t2.out %}
! RUN: %if flangenzyme %{ %fc -cpp -DROUTE1 %flangDirectives -O2 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t2r && %t2r > %t2r.out && diff %t2.out %t2r.out %}

! The nonlinear solve of ../ReverseMode/fixed_point.f90, with the loop marked
! by !$enzyme fixed_point instead of written as a call of enzyme_fixed_point
! with a step function (route 1, with -DROUTE1): the plugin lowers the
! directive to __enzyme_set_fixed_point in the loop. The loop tests a flag
! that its body sets, as MITgcm's STREAMICE loop does. The adjoint is iterated
! at the converged state either way, so the gradient must be that of route 1
! to the last bit (the diff), and match that of the plain loop differentiated
! through every iteration, and centred finite differences.
!
! %flangDirectives and %flangEnzymePlugin each -load a plugin into flang,
! which keeps them both (flang used to keep only the last -load).

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

! Differentiated through every iteration.
subroutine plain(x, y, tol)
  use model
  implicit none
  real(8), intent(in) :: x(m), tol
  real(8), intent(out) :: y
  integer(8) :: i
  logical :: converged
  call init(x)
  i = 0
  converged = .false.
  do while (.not. converged)
    converged = .not. step(i, tol)
    i = i + 1
  end do
  y = loss()
end subroutine plain

! The same loop, iterated to its fixed point in the adjoint.
subroutine fixed(x, y, tol)
#ifdef ROUTE1
  use enzyme
#endif
  use model
  implicit none
  real(8), intent(in) :: x(m), tol
  real(8), intent(out) :: y
  integer(8) :: i
  logical :: converged
  call init(x)
#ifdef ROUTE1
  call enzyme_fixed_point(step, enzyme_fp_state, u, int(8 * m, 8), &
                          enzyme_fp_reduction, 1d-24, &
                          enzyme_fp_control, control, tol)
#else
  i = 0
  converged = .false.
  !$enzyme fixed_point(u) reduction(1d-24) control(control)
  do while (.not. converged)
    converged = .not. step(i, tol)
    i = i + 1
  end do
#endif
  y = loss()
end subroutine fixed

program main
  use enzyme
  use model
  implicit none
  real(8) :: x(m), dx(m), dxp(m), y, dy, yp, ym, fd(m), h, tol
  external :: plain, fixed
  integer :: k, ncontrol_fixed
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

  dx = 0
  dy = 1
  call enzyme_autodiff(fixed, enzyme_dup, x, dx, enzyme_dup, y, dy, &
                       enzyme_const, tol)
  ncontrol_fixed = ncontrol

  dxp = 0
  dy = 1
  call enzyme_autodiff(plain, enzyme_dup, x, dxp, enzyme_dup, y, dy, &
                       enzyme_const, tol)

  ! The same for the directive and route 1, to the last bit.
  print "(a, 5es25.16)", "gradient", dx
  ok = all(abs(dx - dxp) <= 1d-10 * abs(dxp)) .and. &
       all(abs(dx - fd) <= 1d-6 * abs(fd))
  ! The adjoint of the loop was iterated, under the control.
  if (ncontrol_fixed < 3) ok = .false.
  if (.not. ok) then
    print "(a, 5es25.16)", "plain   ", dxp
    print "(a, 5es25.16)", "fd      ", fd
    print *, "control calls", ncontrol_fixed
  end if
  ! CHECK: gradient
  ! CHECK-NEXT: ok
  if (ok) print "(a)", "ok"
end program main
