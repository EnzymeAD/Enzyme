; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false  -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme"  -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %2 = fmul fast double 5.000000e-01, %0
; CHECK-NEXT:   %3 = fdiv fast double %2, %1
; CHECK-NEXT:   %4 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %5 = select fast i1 %4, double 0.000000e+00, double %3
; CHECK-NEXT:   %6 = insertvalue [3 x double] undef, double %5, 0
; CHECK-NEXT:   %7 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %8 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %9 = fmul fast double 5.000000e-01, %7
; CHECK-NEXT:   %10 = fdiv fast double %9, %8
; CHECK-NEXT:   %11 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %12 = select fast i1 %11, double 0.000000e+00, double %10
; CHECK-NEXT:   %13 = insertvalue [3 x double] %6, double %12, 1
; CHECK-NEXT:   %14 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %15 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %16 = fmul fast double 5.000000e-01, %14
; CHECK-NEXT:   %17 = fdiv fast double %16, %15
; CHECK-NEXT:   %18 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %19 = select fast i1 %18, double 0.000000e+00, double %17
; CHECK-NEXT:   %20 = insertvalue [3 x double] %13, double %19, 2
; CHECK-NEXT:   ret [3 x double] %20
; CHECK-NEXT: }