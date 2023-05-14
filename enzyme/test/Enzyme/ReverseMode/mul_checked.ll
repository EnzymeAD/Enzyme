; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -enzyme-strong-zero -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[diffex:.+]] = fmul fast double %[[differet]], %y
; CHECK-NEXT:   %0 = fcmp fast oeq double %differeturn, 0.000000e+00
; CHECK-NEXT:   %[[diffey:.+]] = fmul fast double %[[differet]], %x
; CHECK-NEXT:   %1 = fcmp fast oeq double %differeturn, 0.000000e+00
; CHECK-NEXT:   %2 = select {{(fast )?}}i1 %0, double 0.000000e+00, double %[[diffex]]
; CHECK-NEXT:   %3 = select {{(fast )?}}i1 %1, double 0.000000e+00, double %[[diffey]]
; CHECK-NEXT:   %4 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %5 = insertvalue { double, double } %4, double %3, 1
; CHECK-NEXT:   ret { double, double } %5
