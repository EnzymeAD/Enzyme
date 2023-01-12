; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instcombine -instsimplify -simplifycfg -S | FileCheck %s

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


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %1 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %2 = extractelement <3 x double> %"x'", i64 0
; CHECK-NEXT:   %3 = fmul fast double %2, 5.000000e-01
; CHECK-NEXT:   %4 = fdiv fast double %3, %1
; CHECK-NEXT:   %5 = select {{(fast )?}}i1 %0, double 0.000000e+00, double %4
; CHECK-NEXT:   %6 = insertelement <3 x double> undef, double %5, i64 0
; CHECK-NEXT:   %7 = extractelement <3 x double> %"x'", i64 1
; CHECK-NEXT:   %8 = fmul fast double %7, 5.000000e-01
; CHECK-NEXT:   %9 = fdiv fast double %8, %1
; CHECK-NEXT:   %10 = select {{(fast )?}}i1 %0, double 0.000000e+00, double %9
; CHECK-NEXT:   %11 = insertelement <3 x double> %6, double %10, i64 1
; CHECK-NEXT:   %12 = extractelement <3 x double> %"x'", i64 2
; CHECK-NEXT:   %13 = fmul fast double %12, 5.000000e-01
; CHECK-NEXT:   %14 = fdiv fast double %13, %1
; CHECK-NEXT:   %15 = select {{(fast )?}}i1 %0, double 0.000000e+00, double %14
; CHECK-NEXT:   %16 = insertelement <3 x double> %11, double %15, i64 2
; CHECK-NEXT:   ret <3 x double> %16
; CHECK-NEXT: }