; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @tester(double %x) {
entry:
  %0 = tail call [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define [2 x double] @test_derivative(double %x) {
entry:
  %0 = tail call [2 x double] (...) @__enzyme_fwddiff([2 x double] (double)* nonnull @tester, double %x, double 1.0)
  ret [2 x double] %0
}

declare [2 x double] @__fd_sincos_1(double)

; Function Attrs: nounwind
declare [2 x double] @__enzyme_fwddiff(...)

; CHECK: define internal [2 x double] @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %2 = fmul fast double %1, %"x'"
; CHECK-NEXT:   %3 = insertvalue [2 x double] zeroinitializer, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %5 = fmul fast double %4, %"x'"
; CHECK-NEXT:   %6 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %5
; CHECK-NEXT:   %7 = fadd fast double 0.000000e+00, %6
; CHECK-NEXT:   %8 = insertvalue [2 x double] %3, double %7, 1
; CHECK-NEXT:   ret [2 x double] %8
; CHECK-NEXT: }
