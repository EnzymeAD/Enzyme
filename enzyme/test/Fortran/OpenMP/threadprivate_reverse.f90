! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O2 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -enzyme-globals-default-inactive -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s

! Reverse mode through a parallel region that updates an inactive
! threadprivate module variable (like ICON's timers), accessed through
! __kmpc_threadprivate_cached.

module counters
  implicit none
  public
  integer :: ncalls = 0
  !$omp threadprivate(ncalls)
contains
  subroutine tick()
    ncalls = ncalls + 1
  end subroutine tick
end module counters

module m
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  public
contains
  subroutine f(n, x, y)
    use counters, only: tick
    integer, intent(in) :: n
    real(real64), intent(in) :: x(n)
    real(real64), intent(out) :: y(n)
    integer :: i
    !$omp parallel
    call tick()
    !$omp do
    do i = 1, n
      y(i) = x(i) * x(i)
    end do
    !$omp end do
    !$omp end parallel
  end subroutine f
end module m

program main
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m, only: f
  implicit none

  integer, parameter :: n = 8
  real(real64) :: x(n), dx(n), y(n), dy(n)
  integer :: i

  call omp_set_num_threads(4)
  x = [(0.5d0 * i, i = 1, n)]
  dx(:) = 0
  dy(:) = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)

  ! dy/dx = 2x
  write(*,"(f0.2)") dx(1)
  write(*,"(f0.2)") dx(8)

end program main

! CHECK: 1.00
! CHECK-NEXT: 8.00
