! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O2 -fno-vectorize -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 -fno-vectorize -fopenmp %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a parallel region that captures an inactive integer
! index array through a Fortran POINTER next to active arrays. flang passes
! the captured variables in one struct; Enzyme passes them as separate
! arguments, so that the index loads stay inactive and typed.

module m
contains
  subroutine gather(p_in, edge_idx, p_out, n)
    integer, intent(in) :: n
    real(8), intent(in) :: p_in(4, 3)
    integer, target, intent(in) :: edge_idx(4, 3)
    real(8), intent(inout) :: p_out(4, 3)
    integer, pointer :: iidx(:, :)
    integer :: je, jb
    iidx => edge_idx
    !$omp parallel
    !$omp do private(je) schedule(guided)
    do jb = 1, size(p_out, 2)
      do je = 1, n
        if (iidx(je, jb) >= 1) then
          p_out(je, jb) = p_in(iidx(je, jb), jb)
        end if
      end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine gather
end module m

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m
  implicit none

  integer, parameter :: n = 4, nb = 3
  real(8) :: x(n, nb), dx(n, nb), y(n, nb), dy(n, nb)
  integer :: idx(n, nb), i, j

  call omp_set_num_threads(2)
  do j = 1, nb
    do i = 1, n
      idx(i, j) = n + 1 - i
      x(i, j) = i + 10 * j
    end do
  end do
  dx(:, :) = 0
  y(:, :) = 0
  do i = 1, n
    dy(i, :) = i
  end do
  call enzyme_autodiff(gather, enzyme_dup, x, dx, enzyme_const, idx, &
                       enzyme_dup, y, dy, enzyme_const, n)

  ! y(i, j) = x(n + 1 - i, j), so dx(i, j) = dy(n + 1 - i, j) = n + 1 - i
  write(*,"(4f5.1)") dx(:, 1)
  write(*,"(4f5.1)") dx(:, 3)

end program main

! CHECK: 4.0  3.0  2.0  1.0
! CHECK-NEXT: 4.0  3.0  2.0  1.0
