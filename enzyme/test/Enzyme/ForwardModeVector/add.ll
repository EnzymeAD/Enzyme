; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

define double @tester(double %x, double %y) {
entry:
  %add = fadd double %x, %y
  ret double %add
}

define %struct.Gradients @test_derivative(double %x, double %y){
entry:
  %call = call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.000000e+00, double 0.000000e+00, double %y, double 0.000000e+00, double 1.000000e+00)
  ret %struct.Gradients %call
}

declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %1 = insertelement <2 x double> undef, double %0, i64 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %3 = insertelement <2 x double> %1, double %2, i64 1
; CHECK-NEXT:   %4 = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %5 = insertelement <2 x double> undef, double %4, i64 0
; CHECK-NEXT:   %6 = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %7 = insertelement <2 x double> %5, double %6, i64 1
; CHECK-NEXT:   %8 = fadd fast <2 x double> %3, %7
; CHECK-NEXT:   %9 = extractelement <2 x double> %8, i64 0
; CHECK-NEXT:   %10 = insertvalue [2 x double] undef, double %9, 0
; CHECK-NEXT:   %11 = extractelement <2 x double> %8, i64 1
; CHECK-NEXT:   %12 = insertvalue [2 x double] %10, double %11, 1
; CHECK-NEXT:   ret [2 x double] %12
; CHECK-NEXT: }