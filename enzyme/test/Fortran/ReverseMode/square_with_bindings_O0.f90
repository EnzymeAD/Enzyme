! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s

! NOTE: This test is only configured to run with the flang compiler at -O0
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

program main
  use enzyme, only: enzyme_dup, enzyme_autodiff
  implicit none
  real :: x, dx

  ! Test without an activity descriptor
  x = 3
  print *, square(x)
  dx = 0
  call enzyme_autodiff(square, x, dx)
  print *, dx

  ! Test with an activity descriptor
  x = 4
  print *, square(x)
  dx = 0
  call enzyme_autodiff(square, enzyme_dup, x, dx)
  print *, dx

contains

  real function square(x)
    real, intent(in) :: x
    square = x**2
  end function

end program main

! CHECK: 9
! CHECK-NEXT: 6
! CHECK-NEXT: 16
! CHECK-NEXT: 8
