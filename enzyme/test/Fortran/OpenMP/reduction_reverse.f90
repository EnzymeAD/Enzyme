! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O3 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a sum reduction (__kmpc_reduce / __kmpc_end_reduce).
! With more than four threads the runtime combines the private copies in a
! tree, calling the reduction function itself; 8 threads exercise that.

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  implicit none

  integer, parameter :: n = 16
  real(8) :: x(n), dx(n), s, ds
  integer :: i

  call omp_set_num_threads(8)
  do i = 1, n
    x(i) = 0.5d0 * i
  end do
  dx(:) = 0
  ds = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, s, ds)

  ! d/dx sum(x**2) = 2x
  write(*,"(f0.2)") dx(1)
  write(*,"(f0.2)") dx(2)
  write(*,"(f0.2)") dx(15)
  write(*,"(f0.2)") dx(16)
  write(*,"(f0.2)") sum(dx)

contains

  subroutine f(n, x, s)
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
  end subroutine f

end program main

! CHECK: 1.00
! CHECK-NEXT: 2.00
! CHECK-NEXT: 15.00
! CHECK-NEXT: 16.00
! CHECK-NEXT: 136.00
