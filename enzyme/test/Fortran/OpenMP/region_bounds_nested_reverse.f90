! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O2 -fopenmp -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -enzyme-globals-default-inactive -o %t.ll && %fc -flto -O2 -fopenmp %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O1 -fopenmp %loadFortran %loadFlangEnzyme -mllvm -enzyme-global-activity -mllvm -enzyme-globals-default-inactive %s -o %t2 && %t2 | FileCheck %s %}

! Reverse mode through a worksharing loop whose bounds are set through nested
! IFs in the parallel region, each branch with a worksharing loop of its own
! (as in ICON's mo_nh_diffusion).

module m

contains
  subroutine f(n, lbs, ubs, a, b, x, y, z)
    integer, intent(in) :: n, lbs(3), ubs(3)
    logical, intent(in) :: a, b
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n), z(n)
    integer :: jb, is, ie
!$omp parallel private(is, ie)
    is = lbs(1)
    ie = ubs(1)
    if (a) then
      if (b) then
        is = lbs(2)
        ie = ubs(2)
!$omp do
        do jb = is, ie
          z(jb) = z(jb) + x(jb)
        end do
!$omp end do
      else
        is = lbs(3)
        ie = ubs(3)
!$omp do
        do jb = is, ie
          z(jb) = z(jb) - x(jb)
        end do
!$omp end do
      end if
    end if
!$omp do schedule(dynamic, 1)
    do jb = is, ie
      y(jb) = sin(x(jb)) * x(jb)
    end do
!$omp end do
!$omp end parallel
  end subroutine
end module
program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use omp_lib, only: omp_set_num_threads
  use m
  implicit none
  integer, parameter :: n = 8
  real(8) :: x(n), dx(n), y(n), dy(n), z(n), dz(n)
  integer :: i
  logical :: aa(3) = [.false., .true., .true.], bb(3) = [.false., .true., .false.]
  call omp_set_num_threads(4)
  x = [(0.1d0 * i, i = 1, n)]
  do i = 1, 3
    y = 0; z = 0; dx = 0; dy = 1; dz = 0
    call enzyme_autodiff(f, enzyme_const, n, enzyme_const, [1, 2, 3], enzyme_const, [8, 7, 6], &
                         enzyme_const, aa(i), enzyme_const, bb(i), &
                         enzyme_dup, x, dx, enzyme_dup, y, dy, enzyme_dup, z, dz)
    write(*,"(8f7.4)") dx
  end do
end program

! d/dx (sin(x) x) = cos(x) x + sin(x), on iterations 1-8, 2-7 and 3-6
! CHECK:  0.1993 0.3947 0.5821 0.7578 0.9182 1.0598 1.1796 1.2747
! CHECK-NEXT:  0.0000 0.3947 0.5821 0.7578 0.9182 1.0598 1.1796 0.0000
! CHECK-NEXT:  0.0000 0.0000 0.5821 0.7578 0.9182 1.0598 0.0000 0.0000
