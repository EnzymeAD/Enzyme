! End goal: `flang` with the Enzyme plugin compiles + links a Fortran program
! that uses enzyme_fwddiff into a working executable computing the derivative.
! d/dx (x*x) at x=3 with dx=1 is 2*x*dx = 6. The plugin lowers and differentiates
! the hook entirely inside flang's own pipeline.
!
! REQUIRES: fir_enzyme_flang_plugin
! RUN: %flang_enzyme_driver %s -o %t
! RUN: %t | FileCheck %s

module marks
  integer, bind(C, name="enzyme_dup") :: enzyme_dup
end module

real function square(x)
  real, intent(in) :: x
  square = x * x
end function

program main
  use marks
  real, external :: square
  real, external :: f__enzyme_fwddiff
  real :: x, dx, d
  x = 3.0
  dx = 1.0
  d = f__enzyme_fwddiff(square, enzyme_dup, x, dx)
  print '(F4.1)', d
end program

! CHECK: 6.0
