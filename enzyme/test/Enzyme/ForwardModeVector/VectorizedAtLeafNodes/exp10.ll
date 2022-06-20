; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -O3 -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @exp10(double %x)
  ret double %0
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 2.5>)
  ret <2 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @exp10(double)


; CHECK: define <2 x double> @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @exp10(double %x) #0
; CHECK-NEXT:   %1 = fmul fast double %0, 0x40026BB1BBB55516
; CHECK-NEXT:   %2 = insertelement <2 x double> undef, double %1, i32 0
; CHECK-NEXT:   %3 = fmul fast double %0, 0x4017069E2AA2AA5C
; CHECK-NEXT:   %4 = insertelement <2 x double> %2, double %3, i32 1
; CHECK-NEXT:   ret <2 x double> %4
; CHECK-NEXT: }