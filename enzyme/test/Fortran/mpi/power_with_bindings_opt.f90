! REQUIRES: fortran, mpi
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_dup, enzyme_autodiff, enzyme_fwddiff
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_finalize, &
                 mpi_comm_world, mpi_reduce, mpi_real, mpi_sum
  implicit none

  real :: x, dx, y, dy, s
  integer :: ierr, rank, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Compute the derivatives with reverse mode: 1 (rank 0), 6 (rank 1)
  x = 2.0
  dx = 0.0
  dy = 1.0
  call enzyme_autodiff(power, enzyme_dup, x, dx, enzyme_dup, y, dy)

  ! Take reduction, summing to print on rank 0
  call mpi_reduce(dx, s, 1, mpi_real, mpi_sum, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") s
  end if

  ! Do the same thing with forward mode
  x = 3.0
  dx = 1.0
  dy = 0.0
  call enzyme_fwddiff(power, enzyme_dup, x, dx, enzyme_dup, y, dy)

  ! Take reduction, summing to print on rank 0
  call mpi_reduce(dy, s, 1, mpi_real, mpi_sum, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") s
  end if


  call mpi_finalize(ierr)

contains

  ! Compute the power (rank + 1) of a real
  subroutine power(x, y)
    use mpi, only: mpi_comm_world
    real, intent(in) :: x
    real, intent(out) :: y
    integer :: ierr
    integer :: rank

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    y = x**(rank + 1)
  end subroutine power

end program main

! CHECK: 5.0
! CHECK-NEXT: 7.0
