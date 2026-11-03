! REQUIRES: fortran
! RUN: %fc -flto -O2 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme %enzyme %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Steady 1D Bratu problem
!
!   u'' + lambda * exp(u) = 0 on (0, 1),  u(0) = u(1) = 0,
!
! discretized with second-order central differences and solved by a
! Jacobian-free Newton-Krylov method, after the Bratu example of Ariadne.jl:
! Newton's method on the residual F(u), each Newton system F'(u) d = -F(u)
! solved by conjugate gradients, and every product F'(u) v obtained from Enzyme
! forward mode on the residual routine. The Jacobian is never assembled.
!
! Below the critical lambda the problem has two solutions; with lambda = 1 the
! discrete Jacobian is well conditioned and the iteration converges to the
! lower branch, whose closed form is used as the reference.

module bratu_steady
  implicit none
  private
  public :: dp, residual, jacobian_vector_product
  integer, parameter :: dp = kind(1.0d0)
contains

  ! F(u)_i = (u_{i-1} - 2 u_i + u_{i+1}) / dx**2 + lambda * exp(u_i),
  ! with u_0 = u_{n+1} = 0.
  subroutine residual(n, dx, lambda, u, f)
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
  end subroutine residual

  ! jv = F'(u) v: forward-mode Enzyme on residual, seeded with the tangent v
  ! of u. The tangent of f comes back in jv.
  subroutine jacobian_vector_product(n, dx, lambda, u, v, jv)
    use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
    integer, intent(in) :: n
    real(dp), intent(in) :: dx, lambda
    real(dp), intent(in) :: u(n), v(n)
    real(dp), intent(out) :: jv(n)
    real(dp) :: f(n)
    f = 0.0_dp
    jv = 0.0_dp
    call enzyme_fwddiff(residual, enzyme_const, n, enzyme_const, dx, &
                        enzyme_const, lambda, enzyme_dup, u, v, &
                        enzyme_dup, f, jv)
  end subroutine jacobian_vector_product

end module bratu_steady

program main
  use bratu_steady, only: dp, residual, jacobian_vector_product
  implicit none
  integer, parameter :: n = 200
  real(dp), parameter :: lambda = 1.0_dp
  real(dp), parameter :: pi = 3.141592653589793_dp
  ! Exact lower-branch solution: u(x) = -2 log(cosh(theta (x - 1/2) / 2) / cosh(theta / 4))
  ! with theta the smaller root of theta = sqrt(2 lambda) cosh(theta / 4).
  real(dp), parameter :: theta = 1.5171646_dp
  ! Forcing term of the inexact Newton method: the Krylov solve stops once the
  ! linear residual is below eta times the nonlinear one.
  real(dp), parameter :: eta = 1.0e-3_dp
  real(dp) :: dx, x(n), u(n), f(n), d(n), r(n), p(n), ap(n), reference(n)
  real(dp) :: norm0, norm, tol, alpha, beta, rr, rr_new
  integer :: i, newton, it

  dx = 1.0_dp / (n + 1)
  do i = 1, n
    x(i) = i * dx
  end do
  reference = -2.0_dp * log(cosh(theta * (x - 0.5_dp) / 2.0_dp) / cosh(theta / 4.0_dp))

  ! Initial guess
  u = sin(pi * x)
  call residual(n, dx, lambda, u, f)
  norm0 = sqrt(dot_product(f, f))
  norm = norm0
  tol = 1.0e-8_dp * norm0

  newton = 0
  do while (norm > tol .and. newton < 20)
    ! Conjugate gradients on F'(u) d = -F(u). F'(u) is symmetric, and below
    ! the critical lambda negative definite; CG is indifferent to the sign.
    d = 0.0_dp
    r = -f
    p = r
    rr = dot_product(r, r)
    do it = 1, 10 * n
      call jacobian_vector_product(n, dx, lambda, u, p, ap)
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
    call residual(n, dx, lambda, u, f)
    norm = sqrt(dot_product(f, f))
    newton = newton + 1
  end do

  write(*, "(a,l1)") "converged: ", norm <= tol
  write(*, "(a,f6.4)") "max u: ", maxval(u)
  write(*, "(a,l1)") "matches exact solution: ", maxval(abs(u - reference)) < 1.0e-5_dp
end program main

! CHECK: converged: T
! CHECK-NEXT: max u: 0.1405
! CHECK-NEXT: matches exact solution: T
