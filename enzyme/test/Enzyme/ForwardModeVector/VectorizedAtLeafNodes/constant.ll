; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  ret double 1.000000e+00
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 0.000000e+00, double 1.000000e+00>)
  ret <2 x double> %0
}


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret <2 x double> zeroinitializer
; CHECK-NEXT: }