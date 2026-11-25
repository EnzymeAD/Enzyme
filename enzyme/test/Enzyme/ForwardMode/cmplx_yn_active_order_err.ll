; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S 2>&1 | FileCheck %s

; Differentiating cmplx_yn in forward mode with an active order parameter
; (arg 0, marked AssertingInactiveArg) must emit a structured diagnostic via
; EmitNoDerivativeError instead of silently ignoring arg 0's tangent.

declare [2 x double] @cmplx_yn(double, [2 x double]) readnone nounwind
declare double @__enzyme_fwddiff(...)

define double @active_order(double %n) {
entry:
  %c = call [2 x double] @cmplx_yn(double %n, [2 x double] [double 1.500000e+00, double 0.000000e+00])
  %re = extractvalue [2 x double] %c, 0
  ret double %re
}

define double @test_derivative(double %n) {
entry:
  %d = call double (...) @__enzyme_fwddiff(ptr @active_order, metadata !"enzyme_dup", double %n, double 1.000000e+00)
  ret double %d
}

; CHECK: Enzyme: Unimplemented derivative for argument 0 in forward mode for {{.*}}@cmplx_yn
