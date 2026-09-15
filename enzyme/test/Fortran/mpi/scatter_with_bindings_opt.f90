! Test differentiation through mpi_scatter
!
! REQUIRES: fortran, mpi
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_dup, enzyme_autodiff, enzyme_fwddiff
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_finalize, &
                 mpi_comm_world, mpi_gather, mpi_real
  implicit none

  real :: xg(2), dxg(2), yl, dyl, dyg(2), seed(2)
  integer :: ierr, rank, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Compute the derivatives with forward mode: 1 (rank 0), 4 (rank 1)
  ! The tangent of mpi_scatter still scatters the tangents, so we need to
  ! gather them back to the root process for testing.
  xg = [2.0, 3.0]
  seed = 1.0
  dyl = 0.0
  call enzyme_fwddiff(power, enzyme_dup, xg, seed, enzyme_dup, yl, dyl)
  call mpi_gather(dyl, 1, mpi_real, dyg, 1, mpi_real, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") dyg(1)
    write(*,"(f0.1)") dyg(2)
  end if

  ! Do the same thing with reverse mode: 1 (rank 0), 6 (rank 1)
  ! The adjoint of mpi_scatter gathers the adjoints of the gathered array, so
  ! the root process obtains the derivatives it needs for testing.
  xg = [4.0, 5.0]
  dxg = 0.0
  seed = 1.0
  call enzyme_autodiff(power, enzyme_dup, xg, dxg, enzyme_dup, yl, seed)
  if (rank == 0) then
    write(*,"(f0.1)") dxg(1)
    write(*,"(f0.1)") dxg(2)
  end if

  call mpi_finalize(ierr)

contains

  ! Scatter the input argument then compute its power numprocs
  subroutine power(x, y)
    use mpi, only: mpi_comm_size, mpi_comm_world, mpi_scatter, mpi_real
    real, intent(in) :: x(2)
    real, intent(out) :: y
    real :: xl
    integer :: ierr
    integer :: numprocs

    call mpi_comm_size(mpi_comm_world, numprocs, ierr)
    call mpi_scatter(x, 1, mpi_real, xl, 1, mpi_real, 0, mpi_comm_world, ierr)
    y = xl**numprocs
  end subroutine power

end program main

! CHECK: 4.0
! CHECK-NEXT: 6.0
! CHECK-NEXT: 8.0
! CHECK-NEXT: 10.0
