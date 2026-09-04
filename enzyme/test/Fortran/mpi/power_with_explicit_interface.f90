! REQUIRES: fortran, mpi
! RUN: %fc -flto -O0 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

module power_mod
  implicit none
  public
  interface
    subroutine power__enzyme_autodiff(sr, x_desc, x, dx, y_desc, y, dy)
      implicit none
      interface
        subroutine sr_decal(a, b)
          implicit none
          real, intent(in) :: a
          real, intent(out) :: b
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: x_desc
      real, intent(in) :: x
      real, intent(inout) :: dx
      integer, intent(in) :: y_desc
      real, intent(out) :: y
      real, intent(inout) :: dy
    end subroutine power__enzyme_autodiff
  end interface
  interface
    subroutine power__enzyme_fwddiff(sr, x_desc, x, dx, y_desc, y, dy)
      implicit none
      interface
        subroutine sr_decal(a, b)
          implicit none
          real, intent(in) :: a
          real, intent(out) :: b
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: x_desc
      real, intent(in) :: x
      real, intent(inout) :: dx
      integer, intent(in) :: y_desc
      real, intent(out) :: y
      real, intent(inout) :: dy
    end subroutine power__enzyme_fwddiff
  end interface
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
end module power_mod

program main
  use power_mod, only: power, power__enzyme_autodiff, power__enzyme_fwddiff
  use enzyme, only: enzyme_dup
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_comm_world, &
                 mpi_reduce, mpi_real, mpi_sum, mpi_finalize
  implicit none

  real :: x, dx, y, dy, s
  integer :: rank, ierr, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Compute the derivatives: 1 (rank 0), 6 (rank 1)
  x = 2.0
  dx = 0.0
  dy = 1.0
  call power__enzyme_autodiff(power, enzyme_dup, x, dx, enzyme_dup, y, dy)

  ! Take reduction, summing to print on rank 0
  call mpi_reduce(dx, s, 1, mpi_real, mpi_sum, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") s
  end if

  ! Do the same thing with forward mode
  x = 3.0
  dx = 1.0
  dy = 0.0
  call power__enzyme_fwddiff(power, enzyme_dup, x, dx, enzyme_dup, y, dy)

  ! Take reduction, summing to print on rank 0
  call mpi_reduce(dy, s, 1, mpi_real, mpi_sum, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") s
  end if

  call mpi_finalize(ierr)
end program main

! CHECK: 5.0
! CHECK-NEXT: 7.0
