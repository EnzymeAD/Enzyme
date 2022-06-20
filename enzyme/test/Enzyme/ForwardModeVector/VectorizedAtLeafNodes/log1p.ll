; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -gvn -simplifycfg -instcombine -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call double @log1p(double %x)
  ret double %0
}

define <3 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @log1p(double)


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast double %x, 1.000000e+00
; CHECK-NEXT:   %1 = extractelement <3 x double> %"x'", i32 0
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   %3 = insertelement <3 x double> undef, double %2, i32 0
; CHECK-NEXT:   %4 = extractelement <3 x double> %"x'", i32 1
; CHECK-NEXT:   %5 = fdiv fast double %4, %0
; CHECK-NEXT:   %6 = insertelement <3 x double> %3, double %5, i32 1
; CHECK-NEXT:   %7 = extractelement <3 x double> %"x'", i32 2
; CHECK-NEXT:   %8 = fdiv fast double %7, %0
; CHECK-NEXT:   %9 = insertelement <3 x double> %6, double %8, i32 2
; CHECK-NEXT:   ret <3 x double> %9
; CHECK-NEXT: }