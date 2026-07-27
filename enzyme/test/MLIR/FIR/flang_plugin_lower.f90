! Loading FlangEnzymeMLIR into `flang -fc1` adds the Enzyme passes to flang's own
! HLFIR-to-FIR pipeline, so the enzyme_fwddiff hook call is lowered to an
! enzyme.fwddiff op and differentiated in place, with no separate fir-opt run.
! This checks the resulting FIR; flang_plugin_emit_llvm.f90 checks the LLVM IR.
!
! REQUIRES: flang_enzyme_mlir
! RUN: %flang_enzyme -emit-fir %s -o - | FileCheck %s

module marks
  integer, bind(C, name="enzyme_dup")   :: enzyme_dup
  integer, bind(C, name="enzyme_const") :: enzyme_const
end module

real function square(x, y)
  real, intent(in) :: x, y
  square = x * x + y
end function

subroutine driver(x, dx, y, r)
  use marks
  real, intent(in)  :: x, dx, y
  real, intent(out) :: r
  real, external    :: square
  real, external    :: f__enzyme_fwddiff
  r = f__enzyme_fwddiff(square, enzyme_dup, x, dx, enzyme_const, y)
end subroutine

! The driver now calls the generated dual instead of the hook, and the
! enzyme.fwddiff op has been consumed by differentiation.
! CHECK-LABEL: func.func @_QPdriver
! CHECK: call @fwddiffe_QPsquare(
! CHECK-NOT: enzyme.fwddiff
! CHECK-NOT: fir.call @_QPf__enzyme_fwddiff

! The generated dual computes the tangent of square(x,y) = x*x + y (with x active,
! y constant): d = 2*x*dx.
! CHECK-LABEL: func.func private @fwddiffe_QPsquare
! CHECK: arith.mulf
! CHECK: arith.mulf
! CHECK: arith.addf
