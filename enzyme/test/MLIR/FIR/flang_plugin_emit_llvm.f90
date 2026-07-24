! Loading FlangEnzymeMLIR into `flang -fc1` lowers the Fortran enzyme_fwddiff
! hook to an enzyme.fwddiff op and differentiates it, all inside flang's own
! HLFIR-to-FIR pipeline, so a plain -emit-llvm carries the whole thing through to
! LLVM IR: the hook call is gone and the derivative (d/dx x*x = 2*x) appears as a
! multiply. No separate fir-enzyme-opt invocation.
!
! REQUIRES: fir_enzyme_flang_plugin
! RUN: %flang_enzyme -emit-llvm %s -o - | FileCheck %s

real function square(x)
  real, intent(in) :: x
  square = x * x
end function

module marks
  integer, bind(C, name="enzyme_dup") :: enzyme_dup
end module

subroutine driver(x, dx, r)
  use marks
  real, intent(in)  :: x, dx
  real, intent(out) :: r
  real, external    :: square
  real, external    :: f__enzyme_fwddiff
  r = f__enzyme_fwddiff(square, enzyme_dup, x, dx)
end subroutine

! Once differentiation is wired in, the hook call is gone and the derivative
! (d/dx x*x = 2*x) shows up as a multiply in the LLVM IR.
! CHECK-NOT: f__enzyme_fwddiff
! CHECK: fmul
