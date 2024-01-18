; RUN: if [ %llvmver -gt 12 ]; then if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi; fi
; RUN: if [ %llvmver -gt 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @__enzyme_truncate_value(double, i64, i64)
declare double @__enzyme_expand_value(double, i64, i64)

define double @expand_tester(double %a) {
entry:
  %b = call double @__enzyme_expand_value(double %a, i64 64, i64 32)
  ret double %b
}

define double @truncate_tester(double %a) {
entry:
  %b = call double @__enzyme_truncate_value(double %a, i64 64, i64 32)
  ret double %b
}

; CHECK: define double @expand_tester(double %a) {
; CHECK-DAG: entry:
; CHECK-DAG:   %0 = alloca double, align 8
; CHECK-DAG:   store double %a, ptr %0, align 8
; CHECK-DAG:   %1 = load float, ptr %0, align 4
; CHECK-DAG:   %2 = fpext float %1 to double
; CHECK-DAG:   ret double %2

; CHECK: define double @truncate_tester(double %a) {
; CHECK-DAG: entry:
; CHECK-DAG:   %0 = fptrunc double %a to float
; CHECK-DAG:   %1 = alloca double, align 8
; CHECK-DAG:   store i64 0, ptr %1, align 4
; CHECK-DAG:   store float %0, ptr %1, align 4
; CHECK-DAG:   %2 = load double, ptr %1, align 8
; CHECK-DAG:   ret double %2
