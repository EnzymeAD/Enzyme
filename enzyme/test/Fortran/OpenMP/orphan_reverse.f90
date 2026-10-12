! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O1 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through an orphaned worksharing loop (collapse(2)) in a
! routine called both outside and inside a parallel region. With
! -enzyme-global-activity, the OpenMP runtime calls in it must be known not
! to free memory.

module m
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  public
contains
  subroutine copy2(n, m, src, dest)
    integer, intent(in) :: n, m
    real(real64), intent(in) :: src(n, m)
    real(real64), intent(out) :: dest(n, m)
    integer :: i, j
    !$omp do collapse(2)
    do j = 1, m
      do i = 1, n
        dest(i, j) = src(i, j)
      end do
    end do
    !$omp end do nowait
  end subroutine copy2

  subroutine f(n, m, x, y, z)
    integer, intent(in) :: n, m
    real(real64), intent(in) :: x(n, m)
    real(real64), intent(out) :: y(n, m), z(n, m)
    call copy2(n, m, x * x, y)
    !$omp parallel
    call copy2(n, m, x, z)
    !$omp end parallel
    z = z * y
  end subroutine f
end module m

program main
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m, only: f
  implicit none

  integer, parameter :: n = 3, m = 5
  real(real64) :: x(n, m), dx(n, m), y(n, m), dy(n, m), z(n, m), dz(n, m)
  integer :: i

  call omp_set_num_threads(4)
  x = reshape([(0.1d0 * i, i = 1, n * m)], shape(x))
  dx(:, :) = 0
  dy(:, :) = 0
  dz(:, :) = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_const, m, &
                       enzyme_dup, x, dx, enzyme_dup, y, dy, &
                       enzyme_dup, z, dz)

  ! z = x**3, dz/dx = 3x**2
  write(*,"(f0.3)") dx(1, 1)
  write(*,"(f0.3)") dx(1, 5)
  write(*,"(f0.3)") dx(3, 5)

end program main

! CHECK: .030
! CHECK-NEXT: 5.070
! CHECK-NEXT: 6.750
