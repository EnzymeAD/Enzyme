! Test differentiation through mpi_comm_gather
!
! REQUIRES: fortran, mpi
! RUN: %fc -flto -O0 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

module mpiGather
  implicit none
  public
  interface
    subroutine power__enzyme_autodiff(sr, x_desc, x, dx, y_desc, y, dy)
      implicit none
      interface
        subroutine sr_decal(a, b)
          implicit none
          real, intent(in) :: a
          real, intent(out) :: b(2)
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: x_desc
      real, intent(in) :: x
      real, intent(inout) :: dx
      integer, intent(in) :: y_desc
      real, intent(out) :: y(2)
      real, intent(inout) :: dy(2)
    end subroutine power__enzyme_autodiff
  end interface
  interface
    subroutine power__enzyme_fwddiff(sr, x_desc, x, dx, y_desc, y, dy)
      implicit none
      interface
        subroutine sr_decal(a, b)
          implicit none
          real, intent(in) :: a
          real, intent(out) :: b(2)
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: x_desc
      real, intent(in) :: x
      real, intent(inout) :: dx(2)
      integer, intent(in) :: y_desc
      real, intent(out) :: y(2)
      real, intent(inout) :: dy(2)
    end subroutine power__enzyme_fwddiff
  end interface
contains
  ! Compute the power (rank + 1) of a real and gather the local values
  subroutine power(x, yg)
    use mpi, only: mpi_comm_rank, mpi_comm_world, mpi_gather, mpi_real
    real, intent(in) :: x
    real, intent(out) :: yg(2)
    real :: yl
    integer :: ierr
    integer :: rank

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    yl = x**(rank + 1)
    call mpi_gather(yl, 1, mpi_real, yg, 1, mpi_real, 0, mpi_comm_world, ierr)
  end subroutine power
end module mpiGather

program main
  use mpiGather, only: power, power__enzyme_autodiff, power__enzyme_fwddiff
  use enzyme, only: enzyme_dup
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_comm_world, &
                 mpi_gather, mpi_real, mpi_finalize
  implicit none

  real :: x, dxl, dxg(2), y(2), dy(2), seed(2)
  integer :: rank, ierr, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Compute the derivatives with forward mode: 1 (rank 0), 4 (rank 1)
  ! Here the gathered array of derivatives is produced directly by the
  ! differentiated gather on the root process.
  x = 2.0
  seed = 1.0
  dy = 0.0
  call power__enzyme_fwddiff(power, enzyme_dup, x, seed, enzyme_dup, y, dy)
  if (rank == 0) then
    write(*,"(f0.1)") dy(1)
    write(*,"(f0.1)") dy(2)
  end if

  ! Do the same thing with reverse mode: 1 (rank 0), 6 (rank 1)
  ! The adjoint of mpi_gather scatters the adjoints of the gathered array, so
  ! each process obtains the derivative of its own contribution to the gather;
  ! collect these local derivatives on the root process.
  x = 3.0
  dxl = 0.0
  seed = 1.0
  call power__enzyme_autodiff(power, enzyme_dup, x, dxl, enzyme_dup, y, seed)
  call mpi_gather(dxl, 1, mpi_real, dxg, 1, mpi_real, 0, mpi_comm_world, ierr)
  if (rank == 0) then
    write(*,"(f0.1)") dxg(1)
    write(*,"(f0.1)") dxg(2)
  end if

  call mpi_finalize(ierr)
end program main

! CHECK: 1.0
! CHECK-NEXT: 4.0
! CHECK-NEXT: 1.0
! CHECK-NEXT: 6.0
