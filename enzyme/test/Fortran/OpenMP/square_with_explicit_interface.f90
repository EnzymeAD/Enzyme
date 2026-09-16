! REQUIRES: fortran
! RUN: %fc -flto -O0 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

module squareOMP
  implicit none
  public
  interface
    subroutine square__enzyme_autodiff(sr, n_desc, n, &
                                       x_desc, x, dx, &
                                       y_desc, y, dy)
      implicit none
      interface
        subroutine sr_decal(n, a, b)
          implicit none
          integer, intent(in) :: n
          real, intent(in) :: a(n)
          real, intent(out) :: b(n)
        end subroutine sr_decal
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: n_desc
      integer, intent(in) :: n
      integer, intent(in) :: x_desc
      real, intent(in) :: x(n)
      real, intent(inout) :: dx(n)
      integer, intent(in) :: y_desc
      real, intent(out) :: y(n)
      real, intent(inout) :: dy(n)
    end subroutine square__enzyme_autodiff
  end interface
contains
  subroutine square(n, x, y)
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
    integer :: i
    !$omp parallel do
    do i = 1, n
      y(i) = x(i)**2
    end do
  end subroutine square
end module squareOMP

program main
  use squareOMP, only: square, square__enzyme_autodiff
  use enzyme, only: enzyme_const, enzyme_dup
  use omp_lib, only: omp_get_max_threads
  implicit none

  integer, parameter :: n = 4
  real :: x(n), dx(n)
  real :: y(n), dy(n)
  integer :: i

  if (omp_get_max_threads() < 2) then
    error stop "This test requires OMP_NUM_THREADS >= 2"
  end if

  x(1) = 23.1
  x(2) = 10.0
  x(3) = 100.0
  x(4) = 3.14

  dx(:) = 0.0
  dy(:) = 1.0
  call square__enzyme_autodiff(square, enzyme_const, n, &
                               enzyme_dup, x, dx, &
                               enzyme_dup, y, dy)

  write(*,"(f0.2)") dx(1)
  write(*,"(f0.2)") dx(2)
  write(*,"(f0.2)") dx(3)
  write(*,"(f0.2)") dx(4)
end program main

! CHECK: 46.20
! CHECK-NEXT: 20.00
! CHECK-NEXT: 200.00
! CHECK-NEXT: 6.28
