! Test vector forward mode through mpi_allreduce and mpi_bcast
!
! REQUIRES: fortran, mpi
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %mpi_include %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll %mpi_libs -o %t1 && mpirun -np 2 %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %mpi_include %loadFlangEnzyme %s %mpi_libs -o %t2 && mpirun -np 2 %t2 | FileCheck %s %}

! NOTE: This test is only configured to run with the flang compiler.
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the different signature for enzyme_width used by ifx.

module mpiVectorForward
  implicit none
  public
  interface
    subroutine reduce_bcast__enzyme_fwddiff(sr, width_desc, width, &
                                            x_desc, x, dx1, dx2, &
                                            y_desc, y, dy1, dy2)
      implicit none
      interface
        subroutine sr_decal(a, b)
          implicit none
          real, intent(in) :: a
          real, intent(out) :: b(2)
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, value, intent(in) :: width_desc
      integer, value, intent(in) :: width
      integer, intent(in) :: x_desc
      real, intent(in) :: x
      real, intent(in) :: dx1, dx2
      integer, intent(in) :: y_desc
      real, intent(out) :: y(2)
      real, intent(inout) :: dy1(2), dy2(2)
    end subroutine reduce_bcast__enzyme_fwddiff
  end interface
contains
  ! y(1): allreduce (sum) of the power (rank + 1) of x
  ! y(2): x of the root process (rank 0), broadcast to all processes
  subroutine reduce_bcast(x, y)
    use mpi, only: mpi_comm_rank, mpi_comm_world, mpi_allreduce, mpi_bcast, &
                   mpi_real, mpi_sum
    real, intent(in) :: x
    real, intent(out) :: y(2)
    real :: yl
    integer :: ierr
    integer :: rank

    call mpi_comm_rank(mpi_comm_world, rank, ierr)
    yl = x**(rank + 1)
    call mpi_allreduce(yl, y(1), 1, mpi_real, mpi_sum, mpi_comm_world, ierr)
    y(2) = x
    call mpi_bcast(y(2), 1, mpi_real, 0, mpi_comm_world, ierr)
  end subroutine reduce_bcast
end module mpiVectorForward

program main
  use mpiVectorForward, only: reduce_bcast, reduce_bcast__enzyme_fwddiff
  use enzyme, only: enzyme_dup, enzyme_width
  use mpi, only: mpi_init, mpi_comm_rank, mpi_comm_size, mpi_comm_world, &
                 mpi_finalize
  implicit none

  real :: x, dx1, dx2, y(2), dy1(2), dy2(2)
  integer :: rank, ierr, numprocs

  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call mpi_comm_size(mpi_comm_world, numprocs, ierr)

  if (numprocs /= 2) then
    error stop "This test runs with 2 MPI processes"
  end if

  ! Two tangents at once. With x = 2, the local derivative of x**(rank + 1)
  ! is 1 (rank 0) and 4 (rank 1); the tangent of mpi_allreduce sums the
  ! local tangents, and the tangent of mpi_bcast is the tangent of the root.
  ! Lane 1 seeds dx = 1 on both processes: 1*1 + 4*1 = 5 and 1.
  ! Lane 2 seeds dx = rank + 2:            1*2 + 4*3 = 14 and 2.
  x = 2.0
  dx1 = 1.0
  dx2 = rank + 2.0
  dy1 = 0.0
  dy2 = 0.0
  call reduce_bcast__enzyme_fwddiff(reduce_bcast, enzyme_width, 2, &
                                    enzyme_dup, x, dx1, dx2, &
                                    enzyme_dup, y, dy1, dy2)
  ! Print on rank 1, which receives the broadcast
  if (rank == 1) then
    write(*,"(f0.1)") y(1)
    write(*,"(f0.1)") y(2)
    write(*,"(f0.1)") dy1(1)
    write(*,"(f0.1)") dy1(2)
    write(*,"(f0.1)") dy2(1)
    write(*,"(f0.1)") dy2(2)
  end if

  call mpi_finalize(ierr)
end program main

! CHECK: 6.0
! CHECK-NEXT: 2.0
! CHECK-NEXT: 5.0
! CHECK-NEXT: 1.0
! CHECK-NEXT: 14.0
! CHECK-NEXT: 2.0
