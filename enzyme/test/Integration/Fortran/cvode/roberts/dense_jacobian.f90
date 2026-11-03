! REQUIRES: fortran, cvode
! RUN: %fc -flto -O2 -c %loadFortran %cvode_include %s -o %t.bc
! RUN: %opt %loadEnzyme %enzyme %t.bc -o %t.ll
! RUN: %fc -flto -O2 %t.ll %cvode_libs -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %cvode_include %s %cvode_libs -o %t2 && %t2 | FileCheck %s %}

! CVODE's Robertson example (cv_roberts_dns_f2003) as written: BDF with the
! dense linear solver and a user-supplied Jacobian. Instead of the hand-coded
! entries of the example, the Jacobian callback fills the matrix one column
! at a time with forward-mode Enzyme, seeding the right-hand side with the
! unit vectors. This is the same contract as DVODE's JAC routine with MF = 21.
!
!   dy1/dt = -k1 y1 + k2 y2 y3
!   dy2/dt =  k1 y1 - k2 y2 y3 - k3 y2**2
!   dy3/dt =                     k3 y2**2
!
! with k = (0.04, 1e4, 3e7), y(0) = (1, 0, 0), output at t = 0.4, 4, ..., 4e10.

module robertson
  use, intrinsic :: iso_c_binding, only: c_int, c_long, c_int64_t, c_double, c_ptr
  use fsundials_core_mod, only: N_Vector, SUNMatrix, FN_VGetArrayPointer
  use fsunmatrix_dense_mod, only: FSUNDenseMatrix_Data
  implicit none
  private
  public :: neq, k1, k2, k3, jacobian, rhs, jac_callback
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

  ! Full Jacobian J(i, j) = d ydot(i) / d y(j), column j from the unit seed e_j.
  subroutine jacobian(n, y, jac)
    integer, intent(in) :: n
    real(c_double), intent(in) :: y(n)
    real(c_double), intent(out) :: jac(n, n)
    real(c_double) :: v(n), column(n)
    integer :: j
    do j = 1, n
      v = 0.0d0
      v(j) = 1.0d0
      call jacobian_vector_product(n, y, v, column)
      jac(:, j) = column
    end do
  end subroutine jacobian

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

  ! CVODE Jacobian callback (CVLsJacFn): load the dense SUNMatrix.
  integer(c_int) function jac_callback(t, sunvec_y, sunvec_f, sunmat_j, user_data, &
                                       sunvec_t1, sunvec_t2, sunvec_t3) result(ierr) bind(C)
    real(c_double), value :: t
    type(N_Vector), intent(inout) :: sunvec_y, sunvec_f, sunvec_t1, sunvec_t2, sunvec_t3
    type(SUNMatrix), intent(inout) :: sunmat_j
    type(c_ptr), value :: user_data
    real(c_double), pointer, contiguous :: y(:)
    real(c_double), pointer, contiguous :: jmat(:, :)
    y => FN_VGetArrayPointer(sunvec_y)
    jmat(1:neq, 1:neq) => FSUNDenseMatrix_Data(sunmat_j)
    call jacobian(int(neq), y, jmat)
    ierr = 0
  end function jac_callback

end module robertson

program main
  use, intrinsic :: iso_c_binding, only: c_int, c_long, c_double, c_ptr, c_associated, c_funloc
  use fsundials_core_mod, only: N_Vector, SUNMatrix, SUNLinearSolver, FSUNContext_Create, &
                                FSUNContext_Free, FSUNLinSolFree, FSUNMatDestroy, FN_VDestroy, SUN_COMM_NULL
  use fcvode_mod, only: FCVodeCreate, FCVodeInit, FCVodeSVtolerances, FCVodeSetLinearSolver, &
                        FCVodeSetJacFn, FCVode, FCVodeGetNumJacEvals, FCVodeFree, &
                        CV_BDF, CV_NORMAL, CV_SUCCESS
  use fnvector_serial_mod, only: FN_VMake_Serial
  use fsunmatrix_dense_mod, only: FSUNDenseMatrix
  use fsunlinsol_dense_mod, only: FSUNLinSol_Dense
  use robertson, only: neq, k1, k2, k3, jacobian, rhs, jac_callback
  implicit none
  integer, parameter :: nout = 12
  real(c_double) :: rtol, t0, tout, tret(1)
  real(c_double) :: yval(neq), avtol(neq), jac(neq, neq), jac_ref(neq, neq)
  integer(c_int) :: retval, iout
  integer(c_long) :: nje(1)
  type(N_Vector), pointer :: sunvec_y, sunvec_av
  type(SUNMatrix), pointer :: sunmat_a
  type(SUNLinearSolver), pointer :: sunls
  type(c_ptr) :: cvode_mem, sunctx

  ! Check the Enzyme-filled Jacobian against the closed form of the example
  ! at a generic state before handing it to the solver.
  yval = [0.9d0, 1.0d-5, 0.1d0]
  call jacobian(int(neq), yval, jac)
  jac_ref(1, :) = [-k1, k2 * yval(3), k2 * yval(2)]
  jac_ref(2, :) = [k1, -k2 * yval(3) - 2.0d0 * k3 * yval(2), -k2 * yval(2)]
  jac_ref(3, :) = [0.0d0, 2.0d0 * k3 * yval(2), 0.0d0]
  write(*, "(a,l1)") "Jacobian matches closed form: ", &
    maxval(abs(jac - jac_ref)) <= 1.0d-12 * maxval(abs(jac_ref))

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

  ! Dense matrix and linear solver, with the Jacobian supplied by jac_callback.
  sunmat_a => FSUNDenseMatrix(neq, neq, sunctx)
  if (.not. associated(sunmat_a)) stop 1
  sunls => FSUNLinSol_Dense(sunvec_y, sunmat_a, sunctx)
  if (.not. associated(sunls)) stop 1
  retval = FCVodeSetLinearSolver(cvode_mem, sunls, sunmat_a)
  if (retval /= 0) stop 1
  retval = FCVodeSetJacFn(cvode_mem, c_funloc(jac_callback))
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

  retval = FCVodeGetNumJacEvals(cvode_mem, nje)
  write(*, "(a,l1)") "Enzyme Jacobians used: ", nje(1) > 0
  write(*, "(a,l1)") "mass conserved: ", abs(sum(yval) - 1.0d0) < 1.0d-8

  call FCVodeFree(cvode_mem)
  retval = FSUNLinSolFree(sunls)
  call FSUNMatDestroy(sunmat_a)
  call FN_VDestroy(sunvec_y)
  call FN_VDestroy(sunvec_av)
  retval = FSUNContext_Free(sunctx)
end program main

! CHECK: Jacobian matches closed form: T
! CHECK-NEXT: t = 4.0E-01  9.852E-01  3.386E-05  1.479E-02
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
! CHECK-NEXT: Enzyme Jacobians used: T
! CHECK-NEXT: mass conserved: T
