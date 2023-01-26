; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s

define double @tester(double %x) {
entry:
  %y = bitcast double %x to i64
  %z = bitcast i64 %y to double
  ret double %z
}

define <2 x double> @test_derivative(double %x) {
entry:
  %call = call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.000000e+00, double 0.000000e+00>)
  ret <2 x double> %call
}

declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret <2 x double> %"x'"
; CHECK-NEXT: }