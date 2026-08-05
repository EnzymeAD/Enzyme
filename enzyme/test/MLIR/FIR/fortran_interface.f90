! End-to-end forward mode through the Fortran interface in Fortran/enzyme.f90:
! enzyme_fwddiff and the activity markers come from `use enzyme` rather than from
! hand-written bind(C)/external declarations, and the whole program is compiled
! and linked by the flang driver with the FlangEnzymeMLIR plugin loaded.
!
! REQUIRES: flang_enzyme_mlir, enzyme_module
! RUN: %flang_enzyme_driver %loadFortran %s -o %t && %t | FileCheck %s

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
  implicit none
  real :: x, dx, c, y, dy

  x  = 3.0
  dx = 1.0
  c  = 5.0
  y  = 0.0
  dy = 0.0

  ! y = x*x + c, so dy = 2*x*dx = 6 with c inactive.
  call enzyme_fwddiff(square, enzyme_dup, x, dx, enzyme_const, c, &
                      enzyme_dup, y, dy)

  print *, int(y)
  print *, int(dy)

contains

  subroutine square(a, b, r)
    real, intent(in)  :: a, b
    real, intent(out) :: r
    r = a * a + b
  end subroutine

end program

! CHECK: 14
! CHECK-NEXT: 6
