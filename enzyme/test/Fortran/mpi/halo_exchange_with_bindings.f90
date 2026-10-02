! Test differentiation through a halo exchange written with the
! Fortran MPI bindings, whose requests are INTEGER handles: mpi_isend +
! mpi_recv + mpi_waitall (MITgcm), and mpi_irecv + mpi_isend + mpi_wait
! (ICON).
!
! REQUIRES: fortran, mpi
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-fortran-mpi-runtime-accumulate -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

module halo
  implicit none
  integer, parameter :: n = 3


contains

  ! y = sum over the local values times the neighbour's values (with the
  ! exchange done by isend + recv + waitall)
  subroutine f_isend_recv(x, y)
    use mpi
    double precision, intent(in) :: x(n)
    double precision, intent(out) :: y
    double precision :: sbuf(n), rbuf(n)
    integer :: rank, other, ierr, req(1)
    integer :: stat(MPI_STATUS_SIZE), stats(MPI_STATUS_SIZE, 1)

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    other = 1 - rank
    sbuf = x * x
    call mpi_isend(sbuf, n, mpi_double_precision, other, 7, &
                   mpi_comm_world, req(1), ierr)
    call mpi_recv(rbuf, n, mpi_double_precision, other, 7, &
                  mpi_comm_world, stat, ierr)
    call mpi_waitall(1, req, stats, ierr)
    y = sum(x * rbuf)
  end subroutine f_isend_recv

  ! The same with irecv + isend + wait
  subroutine f_irecv_isend(x, y)
    use mpi
    double precision, intent(in) :: x(n)
    double precision, intent(out) :: y
    double precision :: sbuf(n), rbuf(n)
    integer :: rank, other, ierr, rreq, sreq
    integer :: stat(MPI_STATUS_SIZE)

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    other = 1 - rank
    sbuf = x * x
    call mpi_irecv(rbuf, n, mpi_double_precision, other, 8, &
                   mpi_comm_world, rreq, ierr)
    call mpi_isend(sbuf, n, mpi_double_precision, other, 8, &
                   mpi_comm_world, sreq, ierr)
    call mpi_wait(rreq, stat, ierr)
    call mpi_wait(sreq, stat, ierr)
    y = sum(x * rbuf)
  end subroutine f_irecv_isend

  ! The same with blocking send + recv (rank 0 sends first)
  subroutine f_send_recv(x, y)
    use mpi
    double precision, intent(in) :: x(n)
    double precision, intent(out) :: y
    double precision :: sbuf(n), rbuf(n)
    integer :: rank, other, ierr
    integer :: stat(MPI_STATUS_SIZE)

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    other = 1 - rank
    sbuf = x * x
    if (rank == 0) then
      call mpi_send(sbuf, n, mpi_double_precision, other, 9, &
                    mpi_comm_world, ierr)
      call mpi_recv(rbuf, n, mpi_double_precision, other, 9, &
                    mpi_comm_world, stat, ierr)
    else
      call mpi_recv(rbuf, n, mpi_double_precision, other, 9, &
                    mpi_comm_world, stat, ierr)
      call mpi_send(sbuf, n, mpi_double_precision, other, 9, &
                    mpi_comm_world, ierr)
    end if
    y = sum(x * rbuf)
  end subroutine f_send_recv

end module halo

program main
  use mpi
  use halo
  use enzyme, only: enzyme_dup, enzyme_autodiff, enzyme_fwddiff
  implicit none
  double precision :: x(n), dx(n), y, dy
  integer :: rank, numprocs, ierr, j

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)
  if (numprocs /= 2) error stop "This test runs with 2 MPI processes"

  ! x = (1, 2, 3) on rank 0 and (4, 5, 6) on rank 1. With y_r = sum(x_r *
  ! x_o**2), d(y_0 + y_1)/dx_r = x_o**2 + 2 x_r x_o, i.e.
  ! rank 0: 16+8, 25+20, 36+36 = 24, 45, 72; rank 1: 1+8, 4+20, 9+36 = 9, 24, 45
  x = [(dble(j + 3 * rank), j = 1, n)]

  dx = 0
  dy = 1
  call enzyme_autodiff(f_isend_recv, enzyme_dup, x, dx, enzyme_dup, y, dy)
  call report(dx)

  dx = 0
  dy = 1
  call enzyme_autodiff(f_irecv_isend, enzyme_dup, x, dx, enzyme_dup, y, dy)
  call report(dx)

  dx = 0
  dy = 1
  call enzyme_autodiff(f_send_recv, enzyme_dup, x, dx, enzyme_dup, y, dy)
  call report(dx)

  ! Forward mode with dx = 1: dy_r = sum(x_o**2 + 2 x_r x_o), i.e. 141 on
  ! rank 0 and 78 on rank 1
  dx = 1
  dy = 0
  call enzyme_fwddiff(f_isend_recv, enzyme_dup, x, dx, enzyme_dup, y, dy)
  call report_scalar(dy)

  dx = 1
  dy = 0
  call enzyme_fwddiff(f_irecv_isend, enzyme_dup, x, dx, enzyme_dup, y, dy)
  call report_scalar(dy)

  call mpi_finalize(ierr)

contains

  subroutine report_scalar(dy)
    double precision, intent(in) :: dy
    double precision :: dyg(2)
    call mpi_gather(dy, 1, mpi_double_precision, dyg, 1, &
                    mpi_double_precision, 0, mpi_comm_world, ierr)
    if (rank == 0) write(*, "(2(f0.1,1x))") dyg
  end subroutine report_scalar


  ! Print the derivatives of both processes on the root
  subroutine report(dx)
    double precision, intent(in) :: dx(n)
    double precision :: dxg(n, 2)
    call mpi_gather(dx, n, mpi_double_precision, dxg, n, &
                    mpi_double_precision, 0, mpi_comm_world, ierr)
    if (rank == 0) write(*, "(6(f0.1,1x))") dxg
  end subroutine report
end program main

! CHECK: 24.0 45.0 72.0 9.0 24.0 45.0
! CHECK-NEXT: 24.0 45.0 72.0 9.0 24.0 45.0
! CHECK-NEXT: 24.0 45.0 72.0 9.0 24.0 45.0
! CHECK-NEXT: 141.0 78.0
! CHECK-NEXT: 141.0 78.0
