; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
  ret double %0
}

define <2 x double> @test_derivative(double %x, double %y) {
entry:
  %0 = tail call <2 x double> (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 0.0>, double %y, <2 x double> <double 0.0, double 1.0>)
  ret <2 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'", double %y, <2 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fsub fast double %y, 1.000000e+00
; CHECK-NEXT:   %1 = call fast double @llvm.pow.f64(double %x, double %0)
; CHECK-NEXT:   %2 = fmul fast double %y, %1
; CHECK-NEXT:   %.splatinsert = insertelement <2 x double> poison, double %2, i32 0
; CHECK-NEXT:   %.splat = shufflevector <2 x double> %.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %3 = fmul fast <2 x double> %.splat, %"x'"
; CHECK-NEXT:   %4 = call fast double @llvm.pow.f64(double %x, double %y)
; CHECK-NEXT:   %5 = call fast double @llvm.log.f64(double %x)
; CHECK-NEXT:   %6 = fmul fast double %4, %5
; CHECK-NEXT:   %.splatinsert1 = insertelement <2 x double> poison, double %6, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <2 x double> %.splatinsert1, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %7 = fmul fast <2 x double> %.splat2, %"y'"
; CHECK-NEXT:   %8 = fadd fast <2 x double> %3, %7
; CHECK-NEXT:   ret <2 x double> %8
; CHECK-NEXT: }