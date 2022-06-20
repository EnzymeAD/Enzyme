; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes  -O3 -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.log10.f64(double %x)
  ret double %0
}

define <3 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log10.f64(double)


; CHECK: define <3 x double> @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, 0x40026BB1BBB55516
; CHECK-NEXT:   %.splatinsert.i = insertelement <3 x double> poison, double %0, i32 0
; CHECK-NEXT:   %.splat.i = shufflevector <3 x double> %.splatinsert.i, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %1 = fdiv fast <3 x double> <double 1.000000e+00, double 2.000000e+00, double 3.000000e+00>, %.splat.i
; CHECK-NEXT:   ret <3 x double> %1
; CHECK-NEXT: }