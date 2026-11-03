! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O3 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through worksharing loops with dynamic and guided schedules,
! which Enzyme runs with the static schedule (__kmpc_dispatch_init/next
! become __kmpc_for_static_init), and through a master construct.

program main
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  implicit none

  integer, parameter :: n = 16
  real(real64) :: x(n), dx(n), y(n), dy(n), s, ds
  integer :: i

  call omp_set_num_threads(4)
  do i = 1, n
    x(i) = 0.5d0 * i
  end do
  dx(:) = 0
  dy(:) = 1
  ds = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy, enzyme_dup, s, ds)

  ! y = sin(x)**2 + x**3 and s = 3 sum(x):
  ! dx = sin(2x) + 3x**2 + 3
  write(*,"(f0.4)") dx(1)
  write(*,"(f0.4)") dx(2)
  write(*,"(f0.4)") dx(16)

contains

  subroutine f(n, x, y, s)
    integer, intent(in) :: n
    real(real64), intent(in) :: x(n)
    real(real64), intent(out) :: y(n), s
    integer :: i
    real(real64) :: t
    !$omp parallel private(t)
    !$omp master
    s = 3 * sum(x)
    !$omp end master
    !$omp do schedule(dynamic, 1)
    do i = 1, n
      t = sin(x(i))
      y(i) = t * t
    end do
    !$omp end do
    !$omp do schedule(guided)
    do i = 1, n
      y(i) = y(i) + x(i)**3
    end do
    !$omp end do
    !$omp end parallel
  end subroutine f

end program main

! CHECK: 4.5915
! CHECK-NEXT: 6.9093
! CHECK-NEXT: 194.7121
