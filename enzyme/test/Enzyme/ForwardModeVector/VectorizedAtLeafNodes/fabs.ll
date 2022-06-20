; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.fabs.f64(double %x)
  ret double %0
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 2.0>)
  ret <2 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double)


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %1 = select fast i1 %0, double -1.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %.splatinsert = insertelement <2 x double> poison, double %1, i32 0
; CHECK-NEXT:   %.splat = shufflevector <2 x double> %.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %2 = fmul fast <2 x double> %.splat, %"x'"
; CHECK-NEXT:   ret <2 x double> %2
; CHECK-NEXT: }