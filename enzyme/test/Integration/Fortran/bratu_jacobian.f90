! REQUIRES: fortran
! RUN: %fc -flto -O2 -c %loadFortran %s -o %t.bc
! RUN: %opt %loadEnzyme %enzyme %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Assembling the Jacobian a stiff ODE solver asks for, with Enzyme.
!
! DVODE (jacobwilliams/dvode) with MF = 21 takes a user routine
!
!   subroutine jac(me, neq, t, y, ml, mu, pd, nrowpd)
!
! that loads pd(i, j) with d ydot(i) / d y(j). The routine below has the same
! argument list, minus the solver object, and fills pd one column at a time:
! column j is the forward-mode derivative of the right-hand side seeded with
! the j-th unit vector. The model here is the semi-discrete Bratu equation,
! whose Jacobian is tridiagonal, so the result is checked against the
! closed-form entries and the nonzero count.

module bratu_jac
  implicit none
  private
  public :: dp, dx, lambda, jac
  integer, parameter :: dp = kind(1.0d0)
  ! DVODE would carry these on the solver object (an extension of dvode_t);
  ! module variables stand in for that here.
  real(dp), save :: dx = 0.0_dp
  real(dp), save :: lambda = 0.0_dp
contains

  ! DVODE's f_func: ydot = f(t, y) for the Bratu method-of-lines system.
  subroutine rhs(neq, t, y, ydot)
    integer, intent(in) :: neq
    real(dp), intent(in) :: t
    real(dp), intent(in) :: y(neq)
    real(dp), intent(out) :: ydot(neq)
    real(dp) :: yl, yr
    integer :: i
    do i = 1, neq
      yl = 0.0_dp
      yr = 0.0_dp
      if (i > 1) yl = y(i - 1)
      if (i < neq) yr = y(i + 1)
      ydot(i) = (yl - 2.0_dp * y(i) + yr) / dx**2 + lambda * exp(y(i))
    end do
  end subroutine rhs

  ! DVODE's f_jac for a full Jacobian (MF = 21): pd(i, j) = d ydot(i) / d y(j).
  ! ml and mu are only meaningful for the banded layout (MF = 24) and are
  ! ignored, as DVODE documents for the full case.
  subroutine jac(neq, t, y, ml, mu, pd, nrowpd)
    use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
    integer, intent(in) :: neq, ml, mu, nrowpd
    real(dp), intent(in) :: t
    real(dp), intent(in) :: y(neq)
    real(dp), intent(out) :: pd(nrowpd, neq)
    real(dp) :: v(neq), ydot(neq), column(neq)
    integer :: j
    pd = 0.0_dp
    do j = 1, neq
      v = 0.0_dp
      v(j) = 1.0_dp
      ydot = 0.0_dp
      column = 0.0_dp
      call enzyme_fwddiff(rhs, enzyme_const, neq, enzyme_const, t, &
                          enzyme_dup, y, v, enzyme_dup, ydot, column)
      pd(1:neq, j) = column
    end do
  end subroutine jac

end module bratu_jac

program main
  use bratu_jac, only: dp, dx, lambda, jac
  implicit none
  integer, parameter :: neq = 200
  real(dp), parameter :: pi = 3.141592653589793_dp
  real(dp) :: t, y(neq), pd(neq, neq), pd_ref(neq, neq)
  integer :: i

  dx = 1.0_dp / (neq + 1)
  lambda = 1.0_dp
  t = 0.0_dp
  do i = 1, neq
    y(i) = 0.5_dp * sin(pi * i * dx)
  end do

  call jac(neq, t, y, 0, 0, pd, neq)

  ! Closed-form tridiagonal Jacobian of the Bratu right-hand side.
  pd_ref = 0.0_dp
  do i = 1, neq
    pd_ref(i, i) = -2.0_dp / dx**2 + lambda * exp(y(i))
    if (i > 1) pd_ref(i, i - 1) = 1.0_dp / dx**2
    if (i < neq) pd_ref(i, i + 1) = 1.0_dp / dx**2
  end do

  write(*, "(a,i0)") "nonzeros: ", count(pd /= 0.0_dp)
  write(*, "(a,l1)") "matches closed form: ", &
    maxval(abs(pd - pd_ref)) * dx**2 < 1.0e-12_dp
end program main

! CHECK: nonzeros: 598
! CHECK-NEXT: matches closed form: T
