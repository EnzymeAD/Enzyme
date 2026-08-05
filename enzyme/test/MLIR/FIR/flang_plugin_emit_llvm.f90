! With FlangEnzymeMLIR loaded, a plain -emit-llvm carries differentiation all the
! way to LLVM IR: the enzyme_fwddiff hook call is gone and the derivative
! (d/dx x*x = 2*x) appears as a multiply.
!
! REQUIRES: flang_enzyme_mlir
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
