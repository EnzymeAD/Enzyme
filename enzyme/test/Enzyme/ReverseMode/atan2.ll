; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %y, double %x) {
entry:
  %call = call double @atan2(double %y, double %x)
  ret double %call
}

define double @tester2(double %y) {
entry:
  %call = call double @atan2(double %y, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %y, double %x)
  %1 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @tester2, double %y)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %y, double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:    %[[a0:.+]] = fmul fast double %y, %y
; CHECK-DAG:    %[[a1:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %2 = fadd fast double %[[a1]], %[[a0]]
; CHECK-NEXT:   %3 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %4 = fdiv fast double %3, %2
; CHECK-NEXT:   %5 = fmul fast double %differeturn, %y
; CHECK-NEXT:   %6 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %5
; CHECK-NEXT:   %7 = fdiv fast double %6, %2
; CHECK-NEXT:   %8 = insertvalue { double, double } undef, double %4, 0
; CHECK-NEXT:   %9 = insertvalue { double, double } %8, double %7, 1
; CHECK-NEXT:   ret { double, double } %9
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetester2(double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %y, %y
; CHECK-NEXT:   %1 = fadd fast double 4.000000e+00, %0
; CHECK-NEXT:   %2 = fmul fast double %differeturn, 2.000000e+00
; CHECK-NEXT:   %3 = fdiv fast double %2, %1
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }