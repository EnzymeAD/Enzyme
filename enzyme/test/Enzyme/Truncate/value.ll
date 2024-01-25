; RUN: if [ %llvmver -gt 12 ]; then if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi; fi
; RUN: if [ %llvmver -gt 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @__enzyme_truncate_value(double, i64, i64)
declare double @__enzyme_expand_value(double, i64, i64)

define double @expand_tester(double %a, double * %c) {
entry:
  %b = call double @__enzyme_expand_value(double %a, i64 64, i64 32)
  ret double %b
}

define double @truncate_tester(double %a) {
entry:
  %b = call double @__enzyme_truncate_value(double %a, i64 64, i64 32)
  ret double %b
}

; CHECK: define double @expand_tester(double %a, double* %c) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca double, align 8
; CHECK-NEXT:   store double %a, double* %0, align 8
; CHECK-NEXT:   %1 = bitcast double* %0 to float*
; CHECK-NEXT:   %2 = load float, float* %1, align 4
; CHECK-NEXT:   %3 = fpext float %2 to double
; CHECK-NEXT:   ret double %3

; CHECK: define double @truncate_tester(double %a) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fptrunc double %a to float
; CHECK-NEXT:   %1 = alloca double, align 8
; CHECK-NEXT:   %2 = bitcast double* %1 to i64*
; CHECK-NEXT:   store i64 0, i64* %2, align 4
; CHECK-NEXT:   %3 = bitcast double* %1 to float*
; CHECK-NEXT:   store float %0, float* %3, align 4
; CHECK-NEXT:   %4 = load double, double* %1, align 8
; CHECK-NEXT:   ret double %4
