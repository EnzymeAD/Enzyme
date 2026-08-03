! REQUIRES: fortran
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

module squareReverse
  implicit none
  interface
    subroutine square__enzyme_autodiff(fn, x, dx)
      implicit none
      interface
        real function fn_decal(a)
          implicit none
          real, intent(in) :: a
        end function fn_decal
      end interface
      procedure(fn_decal) :: fn
      real, intent(in) :: x
      real, intent(inout) :: dx
    end subroutine square__enzyme_autodiff
  end interface
contains
  real function square(x)
    real, intent(in) :: x
    square = x**2
  end function square
end module squareReverse

program main
  use squareReverse, only: square, square__enzyme_autodiff
  implicit none
  real :: x, dx

  x = 3
  print *, square(x)

  dx = 0
  call square__enzyme_autodiff(square, x, dx)

  print *, dx
end program main

! CHECK: 9
! CHECK-NEXT: 6
