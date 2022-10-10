; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false  -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme"  -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.exp2.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp2.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %1 = call fast double @llvm.exp2.f64(double %x)
; CHECK-NEXT:   %2 = fmul fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %2, 0x3FE62E42FEFA39EF
; CHECK-NEXT:   %4 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"x'de", align 8
; CHECK-NEXT:   %6 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }
