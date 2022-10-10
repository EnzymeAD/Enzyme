; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false  -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme"  -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @exp10(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.5)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @exp10(double)


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @exp10(double %x) #1
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %2 = fmul fast double %1, %0
; CHECK-NEXT:   %3 = insertvalue [2 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %5 = fmul fast double %4, %0
; CHECK-NEXT:   %6 = insertvalue [2 x double] %3, double %5, 1
; CHECK-NEXT:   %7 = extractvalue [2 x double] %6, 0
; CHECK-NEXT:   %8 = fmul fast double %7, 0x40026BB1BBB55516
; CHECK-NEXT:   %9 = insertvalue [2 x double] undef, double %8, 0
; CHECK-NEXT:   %10 = extractvalue [2 x double] %6, 1
; CHECK-NEXT:   %11 = fmul fast double %10, 0x40026BB1BBB55516
; CHECK-NEXT:   %12 = insertvalue [2 x double] %9, double %11, 1
; CHECK-NEXT:   ret [2 x double] %12
; CHECK-NEXT: }
