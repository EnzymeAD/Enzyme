; RUN: opt < %s -lower-autodiff -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @llvm.autodiff.p0f_f64f64f64f(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind
declare double @llvm.autodiff.p0f_f64f64f64f(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fdiv fast double %x, %y
; CHECK-NEXT:   %diffex = fdiv fast double 1.000000e+00, %y
; CHECK-NEXT:   %1 = fdiv fast double %0, %y
; CHECK-NEXT:   %2 = fsub fast double -0.000000e+00, %1
; CHECK-NEXT:   %3 = insertvalue { double, double } undef, double %diffex, 0
; CHECK-NEXT:   %4 = insertvalue { double, double } %3, double %2, 1
; CHECK-NEXT:   ret { double, double } %4
; CHECK-NEXT: }
