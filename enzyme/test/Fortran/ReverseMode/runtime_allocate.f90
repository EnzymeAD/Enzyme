! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Work arrays allocated and deallocated through LLVM flang's runtime inside
! the differentiated function: a POINTER, and an ALLOCATABLE with STAT=,
! both reallocated on every iteration of a loop.

program main
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup
  implicit none
  integer, parameter :: n = 3
  real(8) :: x(n), dx(n), y, dy

  x = [0.3d0, 0.4d0, 0.5d0]
  dx = 0
  y = 0
  dy = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_dup, y, dy)
  ! y = sum over it = 1..3 of sum_i it * x(i)**2
  write(*,"(3f8.3)") dx

contains

  subroutine f(n, x, y)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y
    real(8), allocatable :: work(:)
    real(8), pointer :: p(:)
    integer :: it, i, ist

    do it = 1, 3
      allocate(work(n), stat=ist)
      allocate(p(n))
      do i = 1, n
        work(i) = x(i) * it
      end do
      do i = 1, n
        p(i) = x(i)
      end do
      do i = 1, n
        y = y + p(i) * work(i)
      end do
      deallocate(p)
      deallocate(work, stat=ist)
    end do
  end subroutine f

end program main

! dy/dx(i) = 2 * (1 + 2 + 3) * x(i)
! CHECK: 3.600   4.800   6.000
