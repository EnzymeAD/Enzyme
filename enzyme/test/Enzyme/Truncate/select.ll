; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define double @f(double %x, double %y, i1 %cond) {
  %res = select i1 %cond, double %x, double %y
  ret double %res
}

declare double (double, double, i1)* @__enzyme_truncate_mem_func(...)
declare double (double, double, i1)* @__enzyme_truncate_op_func(...)

define double @tester(double %x, double %y, i1 %cond) {
entry:
  %ptr = call double (double, double, i1)* (...) @__enzyme_truncate_mem_func(double (double, double, i1)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y, i1 %cond)
  ret double %res
}

define double @tester2(double %x, double %y, i1 %cond) {
entry:
  %ptr = call double (double, double, i1)* (...) @__enzyme_truncate_op_func(double (double, double, i1)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y, i1 %cond)
  ret double %res
}

; CHECK: define double @tester(double %x, double %y, i1 %cond) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = call double @__enzyme_done_truncate_mem_func_64_52to32_23_f(double %x, double %y, i1 %cond)
; CHECK-NEXT:   ret double %res

; CHECK: define internal double @__enzyme_done_truncate_mem_func_64_52to32_23_f(double %x, double %y, i1 %cond) {
; CHECK-DAG:    %1 = alloca double, align 8
; CHECK-DAG:    store double %x, double* %1, align 8
; CHECK-DAG:    %2 = bitcast double* %1 to float*
; CHECK-DAG:    %3 = load float, float* %2, align 4
; CHECK-DAG:    store double %y, double* %1, align 8
; CHECK-DAG:    %4 = bitcast double* %1 to float*
; CHECK-DAG:    %5 = load float, float* %4, align 4
; CHECK-DAG:    %res = select i1 %cond, float %3, float %5
; CHECK-DAG:    %6 = bitcast double* %1 to i64*
; CHECK-DAG:    store i64 0, i64* %6, align 4
; CHECK-DAG:    %7 = bitcast double* %1 to float*
; CHECK-DAG:    store float %res, float* %7, align 4
; CHECK-DAG:    %8 = load double, double* %1, align 8
; CHECK-DAG:    ret double %8

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to32_23_f(double %x, double %y, i1 %cond) {
; CHECK-DAG:   %res = select i1 %cond, double %x, double %y
; CHECK-DAG:   ret double %res
