; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double, double)*, ...)

define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.minnum.f64(double %x, double %y)
  ret double %0
}

define <2 x double> @test_derivative(double %x, double %y) {
entry:
  %0 = tail call <2 x double> (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 0.0>, double %y, <2 x double> <double 0.0, double 1.0>)
  ret <2 x double> %0
}

declare double @llvm.minnum.f64(double, double)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'", double %y, <2 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %1 = select fast i1 %0, <2 x double> %"x'", <2 x double> %"y'"
; CHECK-NEXT:   ret <2 x double> %1
; CHECK-NEXT: }