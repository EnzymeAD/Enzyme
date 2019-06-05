; RUN: opt < %s -lower-autodiff -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @llvm.autodiff.p0f_f64f64f64f(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)

; Function Attrs: nounwind
declare double @llvm.autodiff.p0f_f64f64f64f(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
; CHECK-NEXT:   %1 = fdiv fast double %0, %x
; CHECK-NEXT:   %2 = fmul fast double %1, %y
; CHECK-NEXT:   %3 = call fast double @llvm.log.f64(double %y)
; CHECK-NEXT:   %4 = fmul fast double %0, %3
; CHECK-NEXT:   %5 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %6 = insertvalue { double, double } %5, double %4, 1
; CHECK-NEXT:   ret { double, double } %6
; CHECK-NEXT: }
