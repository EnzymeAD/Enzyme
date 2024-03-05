; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @atanh(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @atanh(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %1 = fmul fast double %x, %x
; CHECK-NEXT:   %2 = fsub fast double 1.000000e+00, %1
; CHECK-NEXT:   %3 = fdiv fast double %0, %2
; CHECK-NEXT:   %4 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"x'de", align 8
; CHECK-NEXT:   %6 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }

