; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s ; fi

declare double @fdim(double, double)

define double @tester(double %x, double %y) {
entry:
  %0 = call double @fdim(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 10.0, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %1 = select fast i1 %0, double 0.000000e+00, double %"x'"
; CHECK-NEXT:   %2 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %3 = fneg fast double %"y'"
; CHECK-NEXT:   %4 = select fast i1 %2, double 0.000000e+00, double %3
; CHECK-NEXT:   %5 = fadd fast double %1, %4
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }


