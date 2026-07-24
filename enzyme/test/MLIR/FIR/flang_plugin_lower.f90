! End-to-end plugin story: loading FlangEnzymeMLIR into `flang -fc1` registers
! the Enzyme HLFIR passes into flang's own HLFIR-to-FIR codegen pipeline (via the
! fir::registerPassPipelineConfigCallback bridge + the HLFIROptEarly extension
! point). During compilation the Fortran enzyme_fwddiff hook call is lowered to
! an enzyme.fwddiff op *and* differentiated in place by the `enzyme` pass -- all
! inside flang, with no separate fir-enzyme-opt invocation. Here we check the FIR
! after differentiation; flang_plugin_emit_llvm.f90 carries it through to LLVM IR.
!
! REQUIRES: fir_enzyme_flang_plugin
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
