; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @acosh(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @acosh(double)

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
; CHECK-NEXT:   %2 = fsub fast double %1, 1.000000e+00
; CHECK-NEXT:   %3 = call fast double @llvm.sqrt.f64(double %2)
; CHECK-NEXT:   %4 = fdiv fast double %0, %3
; CHECK-NEXT:   %5 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %4
; CHECK-NEXT:   store double %6, double* %"x'de", align 8
; CHECK-NEXT:   %7 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %8 = insertvalue { double } undef, double %7, 0
; CHECK-NEXT:   ret { double } %8
; CHECK-NEXT: }

