! Test differentiation through mpi_isend, mpi_wait, mpi_irecv
!
! REQUIRES: fortran, mpi
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s

! NOTE: The in-process FlangEnzyme plugin variant of this test lives in
!       isend_wait_irecv_flangenzyme_O0.f90 (currently XFAIL)

! NOTE: This test is only configured to run with the flang compiler at -O0.
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding.

program main
  use enzyme, only: enzyme_dup, enzyme_autodiff, enzyme_fwddiff
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_finalize, &
                 mpi_comm_world, mpi_real
  implicit none

  real :: x, dx, y, dy
  integer :: ierr, rank, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Compute the derivatives with forward mode
  ! The tangent should send the derivative to the root process.
  x = 2.0
  dx = 1.0
  dy = 0.0
  call enzyme_fwddiff(power, enzyme_dup, x, dx, enzyme_dup, y, dy)
  if (rank == 0) then
    write(*,"(f0.1)") dy
  end if

  ! TODO Do the same thing with reverse mode

  call mpi_finalize(ierr)

contains

  ! Compute the power (rank + 1) of a real on rank 1 and isend it to rank 0
  subroutine power(x, y)
    use mpi, only: mpi_comm_world, mpi_isend, mpi_wait, mpi_irecv, mpi_real, &
                   mpi_status_size
    real, intent(in) :: x
    real, intent(out) :: y
    integer :: ierr, rank, request, status(mpi_status_size)

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    y = x**(rank + 1)
    if (rank == 1) then
      call mpi_isend(y, 1, mpi_real, 0, 0, mpi_comm_world, request, ierr)
      call mpi_wait(request, status, ierr)
    else
      call mpi_irecv(y, 1, mpi_real, 1, 0, mpi_comm_world, request, ierr)
      call mpi_wait(request, status, ierr)
    end if
  end subroutine power

end program main

! CHECK: 4.0
