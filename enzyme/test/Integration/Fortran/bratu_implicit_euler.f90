! REQUIRES: fortran
! RUN: %fc -flto -O2 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme %enzyme %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Time-dependent 1D Bratu problem
!
!   u_t = u_xx + lambda * exp(u) on (0, 1),  u(0, t) = u(1, t) = 0,  u(x, 0) = 0,
!
! discretized in space with central differences (method of lines) and
! integrated with backward Euler. Each implicit step solves
!
!   G(u) = u - u_old - dt * f(u) = 0
!
! with Newton's method, and each Newton system (I - dt f'(u)) d = -G(u) with
! conjugate gradients. The only derivative information needed is the product
! f'(u) v, which Enzyme forward mode produces from the right-hand side routine;
! the solver itself is ordinary Fortran and is not differentiated. This is
! the same division of labour as handing a stiff ODE solver such as DVODE a
! user Jacobian: Enzyme differentiates the model, not the integrator.
!
! For lambda = 1 the solution relaxes to the lower steady state of the Bratu
! equation, whose closed form is used as the reference.

module bratu_time
  implicit none
  private
  public :: dp, implicit_euler_step
  integer, parameter :: dp = kind(1.0d0)
contains

  ! Right-hand side of the semi-discrete system du/dt = f(u):
  ! f(u)_i = (u_{i-1} - 2 u_i + u_{i+1}) / dx**2 + lambda * exp(u_i).
  subroutine rhs(n, dx, lambda, u, f)
    integer, intent(in) :: n
    real(dp), intent(in) :: dx, lambda
    real(dp), intent(in) :: u(n)
    real(dp), intent(out) :: f(n)
    real(dp) :: ul, ur
    integer :: i
    do i = 1, n
      ul = 0.0_dp
      ur = 0.0_dp
      if (i > 1) ul = u(i - 1)
      if (i < n) ur = u(i + 1)
      f(i) = (ul - 2.0_dp * u(i) + ur) / dx**2 + lambda * exp(u(i))
    end do
  end subroutine rhs

  ! jv = f'(u) v by forward-mode Enzyme on rhs.
  subroutine jacobian_vector_product(n, dx, lambda, u, v, jv)
    use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
    integer, intent(in) :: n
    real(dp), intent(in) :: dx, lambda
    real(dp), intent(in) :: u(n), v(n)
    real(dp), intent(out) :: jv(n)
    real(dp) :: f(n)
    f = 0.0_dp
    jv = 0.0_dp
    call enzyme_fwddiff(rhs, enzyme_const, n, enzyme_const, dx, &
                        enzyme_const, lambda, enzyme_dup, u, v, &
                        enzyme_dup, f, jv)
  end subroutine jacobian_vector_product

  ! One backward Euler step: advance u from t to t + dt, Newton-CG on G(u) = 0.
  subroutine implicit_euler_step(n, dx, lambda, dt, u_old, u)
    integer, intent(in) :: n
    real(dp), intent(in) :: dx, lambda, dt
    real(dp), intent(in) :: u_old(n)
    real(dp), intent(inout) :: u(n)
    real(dp), parameter :: eta = 1.0e-3_dp
    real(dp) :: f(n), g(n), d(n), r(n), p(n), jp(n), ap(n)
    real(dp) :: norm0, norm, tol, alpha, beta, rr, rr_new
    integer :: newton, it

    u = u_old
    call rhs(n, dx, lambda, u, f)
    g = u - u_old - dt * f
    norm0 = sqrt(dot_product(g, g))
    norm = norm0
    tol = 1.0e-10_dp * norm0 + 1.0e-14_dp

    newton = 0
    do while (norm > tol .and. newton < 20)
      ! CG on (I - dt f'(u)) d = -G(u); the matrix is symmetric positive
      ! definite whenever -f'(u) is positive semidefinite, which holds below
      ! the critical lambda.
      d = 0.0_dp
      r = -g
      p = r
      rr = dot_product(r, r)
      do it = 1, 10 * n
        call jacobian_vector_product(n, dx, lambda, u, p, jp)
        ap = p - dt * jp
        alpha = rr / dot_product(p, ap)
        d = d + alpha * p
        r = r - alpha * ap
        rr_new = dot_product(r, r)
        if (sqrt(rr_new) <= eta * norm) exit
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
      end do
      u = u + d
      call rhs(n, dx, lambda, u, f)
      g = u - u_old - dt * f
      norm = sqrt(dot_product(g, g))
      newton = newton + 1
    end do
  end subroutine implicit_euler_step

end module bratu_time

program main
  use bratu_time, only: dp, implicit_euler_step
  implicit none
  integer, parameter :: n = 200
  integer, parameter :: nsteps = 50
  real(dp), parameter :: lambda = 1.0_dp
  real(dp), parameter :: dt = 0.05_dp
  ! Lower steady state: u(x) = -2 log(cosh(theta (x - 1/2) / 2) / cosh(theta / 4))
  ! with theta the smaller root of theta = sqrt(2 lambda) cosh(theta / 4).
  real(dp), parameter :: theta = 1.5171646_dp
  real(dp) :: dx, t, x(n), u(n), u_old(n), reference(n)
  integer :: i, step

  dx = 1.0_dp / (n + 1)
  do i = 1, n
    x(i) = i * dx
  end do
  reference = -2.0_dp * log(cosh(theta * (x - 0.5_dp) / 2.0_dp) / cosh(theta / 4.0_dp))

  u = 0.0_dp
  t = 0.0_dp
  do step = 1, nsteps
    u_old = u
    call implicit_euler_step(n, dx, lambda, dt, u_old, u)
    t = t + dt
  end do

  write(*, "(a,f4.1)") "reached t = ", t
  write(*, "(a,f6.4)") "max u: ", maxval(u)
  write(*, "(a,l1)") "matches steady state: ", maxval(abs(u - reference)) < 1.0e-5_dp
end program main

! CHECK: reached t =  2.5
! CHECK-NEXT: max u: 0.1405
! CHECK-NEXT: matches steady state: T
