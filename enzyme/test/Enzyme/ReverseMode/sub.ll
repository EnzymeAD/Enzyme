; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fsub fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = {{(fsub fast double 0.000000e\+00,|fneg fast double)}} %differeturn
; CHECK-NEXT:   %1 = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %2 = insertvalue { double, double } %1, double %0, 1
; CHECK-NEXT:   ret { double, double } %2
; CHECK-NEXT: }
