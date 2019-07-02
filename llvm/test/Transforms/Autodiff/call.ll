; RUN: opt < %s -lower-autodiff -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define dso_local double @add2(double %x) {
entry:
  %add = fadd fast double %x, 2.000000e+00
  ret double %add
}

define dso_local double @add4(double %x) {
entry:
  %call = tail call fast double @add2(double %x)
  %add = fadd fast double %call, 2.000000e+00
  ret double %add
}

define dso_local double @dadd4(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @llvm.autodiff.p0f_f64f64f(double (double)* nonnull @add4, double %x)
  ret double %0
}

declare double @llvm.autodiff.p0f_f64f64f(double (double)*, ...)

; CHECK: define internal { double } @diffeadd4(double %x, double %[[differet:.+]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call fast double @add2(double %x)
; CHECK-NEXT:   %0 = call { double } @diffeadd2(double %x, double %[[differet]])
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeadd2(double %x, double %[[differet:.+]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result:.+]] = insertvalue { double } undef, double %[[differet]], 0
; CHECK-NEXT:   ret { double } %[[result]]
; CHECK-NEXT: }
