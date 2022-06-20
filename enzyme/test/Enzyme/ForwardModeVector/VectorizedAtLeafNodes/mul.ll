; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define <2 x double> @test_derivative(double %x, double %y) {
entry:
  %0 = tail call <2 x double> (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 0.0>, double %y, <2 x double> <double 0.0, double 1.0>)
  ret <2 x double> %0
}


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'", double %y, <2 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.splatinsert = insertelement <2 x double> poison, double %x, i32 0
; CHECK-NEXT:   %.splat = shufflevector <2 x double> %.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert1 = insertelement <2 x double> poison, double %y, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <2 x double> %.splatinsert1, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %0 = fmul fast <2 x double> %"x'", %.splat2
; CHECK-NEXT:   %1 = fmul fast <2 x double> %"y'", %.splat
; CHECK-NEXT:   %2 = fadd fast <2 x double> %0, %1
; CHECK-NEXT:   ret <2 x double> %2
; CHECK-NEXT: }