! REQUIRES: fortran
! RUN: if [[ %fc != ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! NOTE: This test is only configured to run with the flang compiler at -O0
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

program dot
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none

  integer, parameter :: n = 20000000
  integer, parameter :: s = 20
  real :: x(n), y(n), z = 1 / s
  real :: dx(n), dy(n), dz
  integer :: i

  ! Specify input
  do i = 1, n
    x(i) = s / i
    y(i) = s + i - 1 ! NOTE: Minus one to account for 1-indexing
  end do

  ! Compute gradient computation with respect to all variables (x, y, and z)
  dx(:) = 0.0
  dy(:) = 0.0
  dz = 0.0
  call enzyme_autodiff(dot, enzyme_const, n, &
                       enzyme_dup, x, dx, &
                       enzyme_dup, y, dy, &
                       enzyme_dup, z, dz)
  write(*, "(f4.1)") dx(1)
  write(*, "(f4.1)") dy(1)
  write(*, "(f4.1)") dz

  ! Compute gradient computation with respect to just y and z
  dx(:) = 0.0
  dy(:) = 0.0
  dz = 0.0
  call enzyme_autodiff(dot, enzyme_const, n, &
                       enzyme_const, x, &
                       enzyme_dup, y, dy, &
                       enzyme_dup, z, dz)
  write(*, "(f4.1)") dx(1)
  write(*, "(f4.1)") dy(1)
  write(*, "(f4.1)") dz

  ! Compute gradient computation with respect to just z
  dx(:) = 0.0
  dy(:) = 0.0
  dz = 0.0
  call enzyme_autodiff(dot, enzyme_const, n, &
                       enzyme_const, x, &
                       enzyme_const, y, &
                       enzyme_dup, z, dz)
  write(*, "(f4.1)") dx(1)
  write(*, "(f4.1)") dy(1)
  write(*, "(f4.1)") dz

contains

  ! Function for computing the dot product of two vectors and adding a scalar
  real function dot(n, a, b, c)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: a
    real, dimension(n), intent(in) :: b
    real, intent(in) :: c
    integer :: i
    ! TODO: Use the `dot_product` intrinsic.
    !       Requires accounting for `_FortranADotProductReal4` for this to work
    !       at -O0
    ! dot = dot_product(a, b) + c
    dot = c
    do i = 1, n
      dot = dot + a(i) * b(i)
    end do
  end function dot

end program dot

! CHECK: 20.0
! CHECK-NEXT: 20.0
! CHECK-NEXT: 1.0
! CHECK-NEXT: 0.0
! CHECK-NEXT: 20.0
! CHECK-NEXT: 1.0
! CHECK-NEXT: 0.0
! CHECK-NEXT: 0.0
! CHECK-NEXT: 1.0
