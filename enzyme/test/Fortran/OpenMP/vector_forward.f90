! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Vector forward mode (width 3) through a parallel loop with a private
! temporary and a dynamic schedule, and through a sum reduction. The fork
! call passes each lane's shadow as its own argument.

module m
  implicit none
  public
  interface
    subroutine f__enzyme_fwddiff(fn, width_desc, width, n_desc, n, &
                                 x_desc, x, dx1, dx2, dx3, &
                                 y_desc, y, dy1, dy2, dy3, &
                                 s_desc, s, ds1, ds2, ds3)
      implicit none
      interface
        subroutine f_decl(n, x, y, s)
          integer, intent(in) :: n
          real(8), intent(in) :: x(n)
          real(8), intent(out) :: y(n), s
        end subroutine f_decl
      end interface
      procedure(f_decl) :: fn
      integer, value, intent(in) :: width_desc
      integer, value, intent(in) :: width
      integer, intent(in) :: n_desc, n
      integer, intent(in) :: x_desc
      real(8), intent(in) :: x(n), dx1(n), dx2(n), dx3(n)
      integer, intent(in) :: y_desc
      real(8), intent(out) :: y(n)
      real(8), intent(inout) :: dy1(n), dy2(n), dy3(n)
      integer, intent(in) :: s_desc
      real(8), intent(out) :: s
      real(8), intent(inout) :: ds1, ds2, ds3
    end subroutine f__enzyme_fwddiff
  end interface
contains
  subroutine f(n, x, y, s)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n), s
    integer :: i
    real(8) :: t
    !$omp parallel do private(t) schedule(dynamic, 1)
    do i = 1, n
      t = sin(x(i))
      y(i) = t * t
    end do
    !$omp end parallel do
    s = 0
    !$omp parallel do reduction(+:s)
    do i = 1, n
      s = s + x(i)**2
    end do
    !$omp end parallel do
  end subroutine f
end module m

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_width
  use omp_lib, only: omp_set_num_threads
  use m
  implicit none

  integer, parameter :: n = 8
  real(8) :: x(n), y(n), s
  real(8) :: dx1(n), dx2(n), dx3(n), dy1(n), dy2(n), dy3(n), ds1, ds2, ds3
  integer :: i

  call omp_set_num_threads(4)
  x = [(0.5d0 * i, i = 1, n)]
  ! the three directions: e1, e8 and all of x
  dx1(:) = 0
  dx1(1) = 1
  dx2(:) = 0
  dx2(8) = 1
  dx3(:) = 1
  dy1(:) = 0
  dy2(:) = 0
  dy3(:) = 0
  ds1 = 0
  ds2 = 0
  ds3 = 0
  call f__enzyme_fwddiff(f, enzyme_width, 3, enzyme_const, n, &
                         enzyme_dup, x, dx1, dx2, dx3, &
                         enzyme_dup, y, dy1, dy2, dy3, &
                         enzyme_dup, s, ds1, ds2, ds3)

  ! dy/dx = sin(2x), ds/dx = 2x
  write(*,"(f0.4)") dy1(1)
  write(*,"(f0.4)") dy2(8)
  write(*,"(f0.4)") dy3(2)
  write(*,"(f0.2)") ds1
  write(*,"(f0.2)") ds2
  write(*,"(f0.2)") ds3

end program main

! CHECK: .8415
! CHECK-NEXT: .9894
! CHECK-NEXT: .9093
! CHECK-NEXT: 1.00
! CHECK-NEXT: 8.00
! CHECK-NEXT: 36.00
