; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -O3 -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sin.f64(double %x)
  ret double %0
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 2.0>)
  ret <2 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)


; CHECK: define <2 x double> @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.cos.f64(double %x) #2
; CHECK-NEXT:   %.splatinsert.i = insertelement <2 x double> poison, double %0, i32 0
; CHECK-NEXT:   %.splat.i = shufflevector <2 x double> %.splatinsert.i, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %1 = fmul fast <2 x double> %.splat.i, <double 1.000000e+00, double 2.000000e+00>
; CHECK-NEXT:   ret <2 x double> %1
; CHECK-NEXT: }