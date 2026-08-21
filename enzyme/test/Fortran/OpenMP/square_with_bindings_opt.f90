! REQUIRES: fortran
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none

  integer, parameter :: n = 4
  real :: x(n), dx(n)
  real :: y(n), dy(n)
  integer :: i

  x(1) = 23.1
  x(2) = 10.0
  x(3) = 100.0
  x(4) = 3.14

  dx(:) = 0.0
  dy(:) = 1.0
  call enzyme_autodiff(square, enzyme_const, n, &
                       enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)

  write(*,"(f0.2)") dx(1)
  write(*,"(f0.2)") dx(2)
  write(*,"(f0.2)") dx(3)
  write(*,"(f0.2)") dx(4)

contains

  subroutine square(n, x, y)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    real, dimension(n), intent(out) :: y
    integer :: i
    !$omp parallel do
    do i = 1, n
      y(i) = x(i)**2
    end do
    !$omp end parallel do
  end subroutine square

end program main

! CHECK: 46.20
! CHECK-NEXT: 20.00
! CHECK-NEXT: 200.00
! CHECK-NEXT: 6.28
