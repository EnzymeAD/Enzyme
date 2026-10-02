! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O2 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -enzyme-globals-default-inactive -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme -mllvm -enzyme-global-activity -mllvm -enzyme-globals-default-inactive %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a worksharing loop whose bounds are loaded inside the
! parallel region at an index chosen by an IF (as ICON's rl_start), with a
! master construct in one branch. The cache of the loop is sized before the
! fork, where the value merged after the IF becomes a select.

module m
  integer :: nmsg = 0
contains
  subroutine note()
    nmsg = nmsg + 1
  end subroutine note

  subroutine f(n, lbs, ubs, flag, x, y)
    integer, intent(in) :: n, lbs(2), ubs(2)
    logical, intent(in) :: flag
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n)
    integer :: jb, is, ie, rl
    !$omp parallel private(is, ie, rl)
    if (flag) then
      rl = 2
      !$omp master
      call note()
      !$omp end master
    else
      rl = 1
    end if
    is = lbs(rl)
    ie = ubs(rl)
    !$omp do schedule(dynamic, 1)
    do jb = is, ie
      y(jb) = sin(x(jb)) * x(jb)
    end do
    !$omp end do
    !$omp end parallel
  end subroutine f
end module m

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m
  implicit none

  integer, parameter :: n = 8
  real(8) :: x(n), dx(n), y(n), dy(n)
  integer :: i

  call omp_set_num_threads(4)
  x = [(0.1d0 * i, i = 1, n)]

  ! flag: iterations 2 to 7
  y(:) = 0
  dx(:) = 0
  dy(:) = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_const, [1, 2], &
                       enzyme_const, [8, 7], enzyme_const, .true., &
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(8f7.4)") dx

  ! no flag: iterations 1 to 8
  dx(:) = 0
  dy(:) = 1
  call enzyme_autodiff(f, enzyme_const, n, enzyme_const, [1, 2], &
                       enzyme_const, [8, 7], enzyme_const, .false., &
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(8f7.4)") dx

end program main

! d/dx (sin(x) x) = cos(x) x + sin(x)
! CHECK:  0.0000 0.3947 0.5821 0.7578 0.9182 1.0598 1.1796 0.0000
! CHECK-NEXT:  0.1993 0.3947 0.5821 0.7578 0.9182 1.0598 1.1796 1.2747
