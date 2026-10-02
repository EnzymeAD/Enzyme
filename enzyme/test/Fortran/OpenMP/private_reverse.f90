! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O3 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a parallel loop with a private temporary, whose value
! is cached per iteration of each thread's chunk. flang's OpenMPIRBuilder
! computes the chunk's trip count as zext(ub - lb + 1).

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  implicit none

  integer, parameter :: n = 8
  real(8) :: x(n), dx(n), y(n), dy(n)
  integer :: i

  call omp_set_num_threads(4)
  do i = 1, n
    x(i) = 0.5d0 * i
  end do
  dx(:) = 0
  dy(:) = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)

  ! d/dx sin(x)**2 = sin(2x)
  do i = 1, n
    write(*,"(f0.4)") dx(i)
  end do

contains

  subroutine f(n, x, y)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: i
    real(8) :: t
    !$omp parallel do private(t)
    do i = 1, n
      t = sin(x(i))
      y(i) = t * t
    end do
    !$omp end parallel do
  end subroutine f

end program main

! CHECK: .8415
! CHECK-NEXT: .9093
! CHECK-NEXT: .1411
! CHECK-NEXT: -.7568
! CHECK-NEXT: -.9589
! CHECK-NEXT: -.2794
! CHECK-NEXT: .6570
! CHECK-NEXT: .9894
