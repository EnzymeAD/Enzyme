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
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   %"y'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"y'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %1 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %2 = select fast i1 %1, double 0.000000e+00, double %0
; CHECK-NEXT:   %3 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %0
; CHECK-NEXT:   %5 = select fast i1 %1, double %3, double %4
; CHECK-NEXT:   store double %5, double* %"x'de", align 8
; CHECK-NEXT:   %6 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %7 = fneg fast double %0
; CHECK-NEXT:   %8 = select fast i1 %6, double 0.000000e+00, double %7
; CHECK-NEXT:   %9 = load double, double* %"y'de", align 8
; CHECK-NEXT:   %10 = fadd fast double %9, %7
; CHECK-NEXT:   %11 = select fast i1 %6, double %9, double %10
; CHECK-NEXT:   store double %11, double* %"y'de", align 8
; CHECK-NEXT:   %12 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %13 = load double, double* %"y'de", align 8
; CHECK-NEXT:   %14 = insertvalue { double, double } undef, double %12, 0
; CHECK-NEXT:   %15 = insertvalue { double, double } %14, double %13, 1
; CHECK-NEXT:   ret { double, double } %15
; CHECK-NEXT: }

