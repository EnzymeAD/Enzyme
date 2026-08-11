; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %y = tail call fast double @__enzyme_ignore_derivatives(double %x)
  %z = fadd double %y, %x
  %res = fmul double %z, %z
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @__enzyme_ignore_derivatives(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"res'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"res'de", align 8
; CHECK-NEXT:   %"z'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"z'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   %z = fadd double %x, %x
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"res'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"res'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"res'de", align 8
; CHECK-NEXT:   %1 = fmul fast double %0, %z
; CHECK-NEXT:   %2 = load double, double* %"z'de", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"z'de", align 8
; CHECK-NEXT:   %4 = fmul fast double %0, %z
; CHECK-NEXT:   %5 = load double, double* %"z'de", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %4
; CHECK-NEXT:   store double %6, double* %"z'de", align 8
; CHECK-NEXT:   %7 = load double, double* %"z'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"z'de", align 8
; CHECK-NEXT:   %8 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %9 = fadd fast double %8, %7
; CHECK-NEXT:   store double %9, double* %"x'de", align 8
; CHECK-NEXT:   %10 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %11 = insertvalue { double } undef, double %10, 0
; CHECK-NEXT:   ret { double } %11
; CHECK-NEXT: }
