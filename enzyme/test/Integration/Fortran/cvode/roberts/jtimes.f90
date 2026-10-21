! REQUIRES: fortran, cvode
! RUN: %fc -flto -O2 -c %loadFortran %cvode_include %s -o %t.bc
! RUN: %opt %loadEnzyme %enzyme %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll %cvode_libs -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %cvode_include %s %cvode_libs -o %t2 && %t2 | FileCheck %s %}

! CVODE's Robertson example (cv_roberts_dns_f2003), integrated with BDF and a
! matrix-free Newton-Krylov linear solver (SPGMR, no preconditioner), where
! the Jacobian-vector products the Krylov iteration needs come from Enzyme
! forward mode on the right-hand side. CVODE is not differentiated; it only
! calls the jtimes routine, which seeds the model with the tangent v.
!
!   dy1/dt = -k1 y1 + k2 y2 y3
!   dy2/dt =  k1 y1 - k2 y2 y3 - k3 y2**2
!   dy3/dt =                     k3 y2**2
!
! with k = (0.04, 1e4, 3e7), y(0) = (1, 0, 0), output at t = 0.4, 4, ..., 4e10.

module robertson
  use, intrinsic :: iso_c_binding, only: c_int, c_long, c_int64_t, c_double, c_ptr
  use fsundials_core_mod, only: N_Vector, FN_VGetArrayPointer
  implicit none
  private
  public :: neq, rhs, jtimes
  integer(c_int64_t), parameter :: neq = 3
  real(c_double), parameter :: k1 = 0.04d0, k2 = 1.0d4, k3 = 3.0d7
contains

  ! The model, on plain arrays: this is what Enzyme differentiates.
  subroutine rates(n, y, ydot)
    integer, intent(in) :: n
    real(c_double), intent(in) :: y(n)
    real(c_double), intent(out) :: ydot(n)
    ydot(1) = -k1 * y(1) + k2 * y(2) * y(3)
    ydot(3) = k3 * y(2)**2
    ydot(2) = -ydot(1) - ydot(3)
  end subroutine rates

  ! jv = (d rates / d y) v by forward-mode Enzyme, seeded with v.
  subroutine jacobian_vector_product(n, y, v, jv)
    use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
    integer, intent(in) :: n
    real(c_double), intent(in) :: y(n), v(n)
    real(c_double), intent(out) :: jv(n)
    real(c_double) :: ydot(n)
    ydot = 0.0d0
    jv = 0.0d0
    call enzyme_fwddiff(rates, enzyme_const, n, enzyme_dup, y, v, &
                        enzyme_dup, ydot, jv)
  end subroutine jacobian_vector_product

  ! CVODE right-hand side callback (CVRhsFn).
  integer(c_int) function rhs(t, sunvec_y, sunvec_f, user_data) &
    result(ierr) bind(C)
    real(c_double), value :: t
    type(N_Vector), intent(inout) :: sunvec_y, sunvec_f
    type(c_ptr), value :: user_data
    real(c_double), pointer, contiguous :: y(:), f(:)
    y => FN_VGetArrayPointer(sunvec_y)
    f => FN_VGetArrayPointer(sunvec_f)
    call rates(int(neq), y, f)
    ierr = 0
  end function rhs

  ! CVODE Jacobian-times-vector callback (CVLsJacTimesVecFn): Jv = J(t, y) v.
  integer(c_int) function jtimes(sunvec_v, sunvec_jv, t, sunvec_y, sunvec_fy, &
                                 user_data, sunvec_tmp) result(ierr) bind(C)
    type(N_Vector), intent(inout) :: sunvec_v, sunvec_jv, sunvec_y, sunvec_fy, sunvec_tmp
    real(c_double), value :: t
    type(c_ptr), value :: user_data
    real(c_double), pointer, contiguous :: v(:), jv(:), y(:)
    v => FN_VGetArrayPointer(sunvec_v)
    jv => FN_VGetArrayPointer(sunvec_jv)
    y => FN_VGetArrayPointer(sunvec_y)
    call jacobian_vector_product(int(neq), y, v, jv)
    ierr = 0
  end function jtimes

end module robertson

program main
  use, intrinsic :: iso_c_binding, only: c_int, c_long, c_double, c_ptr, c_associated, c_funloc, c_null_funptr
  use fsundials_core_mod, only: N_Vector, SUNMatrix, SUNLinearSolver, FSUNContext_Create, &
                                FSUNContext_Free, FSUNLinSolFree, FN_VDestroy, SUN_COMM_NULL, SUN_PREC_NONE
  use fcvode_mod, only: FCVodeCreate, FCVodeInit, FCVodeSVtolerances, FCVodeSetLinearSolver, &
                        FCVodeSetJacTimes, FCVode, FCVodeGetNumSteps, FCVodeGetNumJtimesEvals, &
                        FCVodeFree, CV_BDF, CV_NORMAL, CV_SUCCESS
  use fnvector_serial_mod, only: FN_VMake_Serial
  use fsunlinsol_spgmr_mod, only: FSUNLinSol_SPGMR
  use robertson, only: neq, rhs, jtimes
  implicit none
  integer, parameter :: nout = 12
  real(c_double) :: rtol, t0, tout, tret(1)
  real(c_double) :: yval(neq), avtol(neq)
  integer(c_int) :: retval, iout
  integer(c_long) :: nst(1), njtimes(1)
  type(N_Vector), pointer :: sunvec_y, sunvec_av
  type(SUNMatrix), pointer :: sunmat_null
  type(SUNLinearSolver), pointer :: sunls
  type(c_ptr) :: cvode_mem, sunctx

  retval = FSUNContext_Create(SUN_COMM_NULL, sunctx)

  yval = [1.0d0, 0.0d0, 0.0d0]
  rtol = 1.0d-8
  avtol = [1.0d-12, 1.0d-18, 1.0d-12]
  t0 = 0.0d0
  tout = 0.4d0

  sunvec_y => FN_VMake_Serial(neq, yval, sunctx)
  sunvec_av => FN_VMake_Serial(neq, avtol, sunctx)
  if (.not. (associated(sunvec_y) .and. associated(sunvec_av))) stop 1

  cvode_mem = FCVodeCreate(CV_BDF, sunctx)
  if (.not. c_associated(cvode_mem)) stop 1
  retval = FCVodeInit(cvode_mem, c_funloc(rhs), t0, sunvec_y)
  if (retval /= 0) stop 1
  retval = FCVodeSVtolerances(cvode_mem, rtol, sunvec_av)
  if (retval /= 0) stop 1

  ! Matrix-free: a Krylov linear solver, no SUNMatrix, and the user jtimes.
  sunls => FSUNLinSol_SPGMR(sunvec_y, SUN_PREC_NONE, 5, sunctx)
  if (.not. associated(sunls)) stop 1
  sunmat_null => null()
  retval = FCVodeSetLinearSolver(cvode_mem, sunls, sunmat_null)
  if (retval /= 0) stop 1
  retval = FCVodeSetJacTimes(cvode_mem, c_null_funptr, c_funloc(jtimes))
  if (retval /= 0) stop 1

  do iout = 1, nout
    retval = FCVode(cvode_mem, tout, sunvec_y, tret(1), CV_NORMAL)
    if (retval /= CV_SUCCESS) then
      print "(a,i0)", "FCVode failed: ", retval
      stop 1
    end if
    write(*, "(a,es8.1,3(1x,es10.3))") "t =", tout, yval
    tout = tout * 10.0d0
  end do

  retval = FCVodeGetNumSteps(cvode_mem, nst)
  retval = FCVodeGetNumJtimesEvals(cvode_mem, njtimes)
  write(*, "(a,l1)") "Enzyme Jacobian-vector products used: ", njtimes(1) > 0
  write(*, "(a,l1)") "mass conserved: ", abs(sum(yval) - 1.0d0) < 1.0d-8

  call FCVodeFree(cvode_mem)
  retval = FSUNLinSolFree(sunls)
  call FN_VDestroy(sunvec_y)
  call FN_VDestroy(sunvec_av)
  retval = FSUNContext_Free(sunctx)
end program main

! CHECK: t = 4.0E-01  9.852E-01  3.386E-05  1.479E-02
! CHECK-NEXT: t = 4.0E+00  9.055E-01  2.240E-05  9.446E-02
! CHECK-NEXT: t = 4.0E+01  7.158E-01  9.186E-06  2.842E-01
! CHECK-NEXT: t = 4.0E+02  4.505E-01  3.223E-06  5.495E-01
! CHECK-NEXT: t = 4.0E+03  1.832E-01  8.942E-07  8.168E-01
! CHECK-NEXT: t = 4.0E+04  3.898E-02  1.622E-07  9.610E-01
! CHECK-NEXT: t = 4.0E+05  4.938E-03  1.985E-08  9.951E-01
! CHECK-NEXT: t = 4.0E+06  5.168E-04  2.068E-09  9.995E-01
! CHECK-NEXT: t = 4.0E+07  5.203E-05  2.081E-10  9.999E-01
! CHECK-NEXT: t = 4.0E+08  5.208E-06  2.083E-11  1.000E+00
! CHECK-NEXT: t = 4.0E+09  5.208E-07  2.083E-12  1.000E+00
! CHECK-NEXT: t = 4.0E+10  5.208E-08  2.083E-13  1.000E+00
! CHECK-NEXT: Enzyme Jacobian-vector products used: T
! CHECK-NEXT: mass conserved: T
