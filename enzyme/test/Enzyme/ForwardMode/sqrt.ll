; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %1 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %2 = fmul fast double 5.000000e-01, %"x'"
; CHECK-NEXT:   %3 = fdiv fast double %2, %1
; CHECK-NEXT:   %4 = select{{( fast)?}} i1 %0, double 0.000000e+00, double %3
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }
