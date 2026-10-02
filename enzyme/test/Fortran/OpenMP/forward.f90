! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Forward mode through a parallel loop with a private temporary and through
! a sum reduction.

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
  use omp_lib, only: omp_set_num_threads
  implicit none

  integer, parameter :: n = 16
  real(8) :: x(n), dx(n), y(n), dy(n), s, ds
  integer :: i

  call omp_set_num_threads(8)
  do i = 1, n
    x(i) = 0.5d0 * i
  end do
  dx(:) = 1
  dy(:) = 0
  call enzyme_fwddiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                      enzyme_dup, y, dy)
  ! d/dx sin(x)**2 = sin(2x)
  write(*,"(f0.4)") dy(1)
  write(*,"(f0.4)") dy(2)
  write(*,"(f0.4)") dy(16)

  s = 0
  ds = 0
  call enzyme_fwddiff(g, enzyme_const, n, enzyme_dup, x, dx, &
                      enzyme_dup, s, ds)
  ! sum(2x)
  write(*,"(f0.2)") ds

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

  subroutine g(n, x, s)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: s
    integer :: i
    s = 0
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + x(i)**2
    end do
    !$omp end parallel do
  end subroutine g

end program main

! CHECK: .8415
! CHECK-NEXT: .9093
! CHECK-NEXT: -.2879
! CHECK-NEXT: 136.00
