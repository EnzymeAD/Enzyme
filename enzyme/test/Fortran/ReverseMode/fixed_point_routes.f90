! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t0 && %t0 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %flangEnzymePlugin -mllvm -enzyme-global-activity=1 %s %linkFortran -o %t2 && %t2 | FileCheck %s %}

! The two ways to mark a fixed-point loop, enzyme_fixed_point on a step
! function and __enzyme_set_fixed_point in the loop (what !$enzyme
! fixed_point lowers to), differentiated in one program, each after the
! other. Both gradients must match AD through every iteration of the plain
! loop.

module model
  use, intrinsic :: iso_fortran_env, only: real64, int64
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
  private
  public :: m, u, step, loss, init, mark, step1
  integer, parameter :: m = 5
  real(real64) :: u(m) = 0, p(m) = 0
  interface
    subroutine mark(reduction, max_iters, control, state, bytes) &
        bind(C, name="__enzyme_set_fixed_point")
      import :: c_double, c_int64_t, c_ptr, real64, m
      implicit none
      real(c_double), value :: reduction
      integer(c_int64_t), value :: max_iters
      type(c_ptr), value :: control
      real(real64), intent(in) :: state(m)
      integer(c_int64_t), value :: bytes
    end subroutine mark
  end interface
contains
  logical function step(tol)
    real(real64), intent(in) :: tol
    real(real64) :: tmp(m), err
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

  integer function step1(i, tol)
    integer(int64), value :: i
    real(real64), intent(in) :: tol
    step1 = 0
    if (step(tol)) step1 = 1
  end function step1

  real(real64) function loss()
    integer :: k
    loss = 0
    do k = 1, m
      loss = loss + u(k)**3
    end do
  end function loss

  subroutine init(x)
    real(real64), intent(in) :: x(m)
    integer :: k
    do k = 1, m
      u(k) = 0
      p(k) = 0.5d0 * x(k)
    end do
  end subroutine init
end module model

program main
  use, intrinsic :: iso_fortran_env, only: real64, int64
  use, intrinsic :: iso_c_binding, only: c_null_ptr
  use enzyme, only: enzyme_autodiff, enzyme_dup, enzyme_const, enzyme_fixed_point, enzyme_fp_state, enzyme_fp_reduction
  use model, only: m, u, step, loss, init, mark, step1
  implicit none
  real(real64) :: x(m), dx(m), d0(m), d1(m), y, dy, tol
  integer :: k
  logical :: ok

  tol = 1d-15
  do k = 1, m
    x(k) = 0.2d0 + 0.1d0 * k
    d0(k) = 0
    d1(k) = 0
    dx(k) = 0
  end do
  dy = 1
  call enzyme_autodiff(plain, enzyme_dup, x, d0, enzyme_dup, y, dy, &
                       enzyme_const, tol)
  ! Each route, then the first again: neither may disturb the other.
  dy = 1
  call enzyme_autodiff(route1, enzyme_dup, x, d1, enzyme_dup, y, dy, &
                       enzyme_const, tol)
  ok = all(abs(d1 - d0) <= 1d-10 * abs(d0))
  dy = 1
  call enzyme_autodiff(fixed, enzyme_dup, x, dx, enzyme_dup, y, dy, &
                       enzyme_const, tol)
  ok = ok .and. all(abs(dx - d0) <= 1d-10 * abs(d0))
  d1 = 0
  dy = 1
  call enzyme_autodiff(route1, enzyme_dup, x, d1, enzyme_dup, y, dy, &
                       enzyme_const, tol)
  ok = ok .and. all(abs(d1 - d0) <= 1d-10 * abs(d0))

  if (.not. ok) then
    print *, "route 1", d1
    print *, "loop", dx
    print *, "expected", d0
  end if
  ! CHECK: ok
  if (ok) print "(a)", "ok"

contains

  subroutine plain(x, y, tol)
    real(real64), intent(in) :: x(m), tol
    real(real64), intent(out) :: y
    call init(x)
    do while (step(tol))
    end do
    y = loss()
  end subroutine plain

  subroutine route1(x, y, tol)
    real(real64), intent(in) :: x(m), tol
    real(real64), intent(out) :: y
    call init(x)
    call enzyme_fixed_point(step1, enzyme_fp_state, u, int(8 * m, int64), &
                            enzyme_fp_reduction, 1d-24, tol)
    y = loss()
  end subroutine route1

  subroutine fixed(x, y, tol)
    real(real64), intent(in) :: x(m), tol
    real(real64), intent(out) :: y
    call init(x)
    do while (step(tol))
      call mark(1d-24, -1_int64, c_null_ptr, u, int(8 * m, int64))
    end do
    y = loss()
  end subroutine fixed
end program main
