! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a routine containing a parallel loop, called twice
! with its input overwritten in between.

module m
contains
  subroutine step(n, x, y)
    integer, intent(in) :: n
    real(8), intent(inout) :: x(n)
    real(8), intent(inout) :: y(n)
    integer :: i
    !$omp parallel do
    do i = 1, n
      y(i) = y(i) + x(i) * x(i)
    end do
    !$omp end parallel do
  end subroutine step

  subroutine drive(n, x, y)
    integer, intent(in) :: n
    real(8), intent(inout) :: x(n)
    real(8), intent(inout) :: y(n)
    call step(n, x, y)
    x = 2 * x
    call step(n, x, y)
  end subroutine drive
end module m

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m
  implicit none

  integer, parameter :: n = 8
  real(8) :: x(n), dx(n), y(n), dy(n)
  integer :: i

  call omp_set_num_threads(4)
  do i = 1, n
    x(i) = 0.5d0 * i
  end do
  y(:) = 0
  dx(:) = 0
  dy(:) = 1
  call enzyme_autodiff(drive, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)

  ! y = x**2 + (2x)**2, dy/dx = 10x
  write(*,"(f0.2)") dx(1)
  write(*,"(f0.2)") dx(2)
  write(*,"(f0.2)") dx(8)

end program main

! CHECK: 5.00
! CHECK-NEXT: 10.00
! CHECK-NEXT: 40.00
