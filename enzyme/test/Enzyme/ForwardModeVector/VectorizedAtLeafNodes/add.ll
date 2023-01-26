; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %add = fadd double %x, %y
  ret double %add
}

define <2 x double> @test_derivative(double %x, double %y) {
entry:
  %call = call <2 x double> (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.000000e+00, double 0.000000e+00>, double %y, <2 x double> <double 0.000000e+00, double 1.000000e+00>)
  ret <2 x double> %call
}

declare <2 x double> @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'", double %y, <2 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast <2 x double> %"x'", %"y'"
; CHECK-NEXT:   ret <2 x double> %0
; CHECK-NEXT: }