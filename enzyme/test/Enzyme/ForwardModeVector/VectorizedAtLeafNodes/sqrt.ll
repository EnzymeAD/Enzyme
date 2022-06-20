; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -O3 -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define <3 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)


; CHECK: define <3 x double> @test_derivative(double %x)
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %1 = tail call fast double @llvm.sqrt.f64(double %x) #2
; CHECK-NEXT:   %2 = insertelement <2 x double> poison, double %1, i32 0
; CHECK-NEXT:   %3 = shufflevector <2 x double> %2, <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:   %4 = fdiv fast <2 x double> <double 5.000000e-01, double 1.000000e+00>, %3
; CHECK-NEXT:   %5 = insertelement <2 x i1> poison, i1 %0, i32 0
; CHECK-NEXT:   %6 = shufflevector <2 x i1> %5, <2 x i1> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:   %7 = select <2 x i1> %6, <2 x double> zeroinitializer, <2 x double> %4
; CHECK-NEXT:   %8 = extractelement <2 x double> %7, i32 0
; CHECK-NEXT:   %9 = insertelement <3 x double> undef, double %8, i32 0
; CHECK-NEXT:   %10 = extractelement <2 x double> %7, i32 1
; CHECK-NEXT:   %11 = insertelement <3 x double> %9, double %10, i32 1
; CHECK-NEXT:   %12 = fdiv fast double 1.500000e+00, %1
; CHECK-NEXT:   %13 = select fast i1 %0, double 0.000000e+00, double %12
; CHECK-NEXT:   %14 = insertelement <3 x double> %11, double %13, i32 2
; CHECK-NEXT:   ret <3 x double> %14
; CHECK-NEXT: }