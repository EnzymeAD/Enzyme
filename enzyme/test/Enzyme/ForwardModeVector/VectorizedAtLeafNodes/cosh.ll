; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @cosh(double %x)
  ret double %0
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 0.000000e+00, double 1.000000e+00>)
  ret <2 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @cosh(double)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @sinh(double %x) #1
; CHECK-NEXT:   %1 = extractelement <2 x double> %"x'", i32 0
; CHECK-NEXT:   %2 = fmul fast double %1, %0
; CHECK-NEXT:   %3 = insertelement <2 x double> undef, double %2, i32 0
; CHECK-NEXT:   %4 = extractelement <2 x double> %"x'", i32 1
; CHECK-NEXT:   %5 = fmul fast double %4, %0
; CHECK-NEXT:   %6 = insertelement <2 x double> %3, double %5, i32 1
; CHECK-NEXT:   ret <2 x double> %6
; CHECK-NEXT: }