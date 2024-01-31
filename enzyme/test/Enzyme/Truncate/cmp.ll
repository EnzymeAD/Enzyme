; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define i1 @f(double %x, double %y) {
  %res = fcmp olt double %x, %y
  ret i1 %res
}

declare i1 (double, double)* @__enzyme_truncate_mem_func(...)
declare i1 (double, double)* @__enzyme_truncate_op_func(...)

define i1 @tester(double %x, double %y) {
entry:
  %ptr = call i1 (double, double)* (...) @__enzyme_truncate_mem_func(i1 (double, double)* @f, i64 64, i64 32)
  %res = call i1 %ptr(double %x, double %y)
  ret i1 %res
}
define i1 @tester_op(double %x, double %y) {
entry:
  %ptr = call i1 (double, double)* (...) @__enzyme_truncate_op_func(i1 (double, double)* @f, i64 64, i64 32)
  %res = call i1 %ptr(double %x, double %y)
  ret i1 %res
}

; CHECK: define i1 @tester(double %x, double %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = call i1 @__enzyme_done_truncate_mem_func_64_52_32_23_f(double %x, double %y)
; CHECK-NEXT:   ret i1 %res

; CHECK: define internal i1 @__enzyme_done_truncate_mem_func_64_52_32_23_f(double %x, double %y) {
; CHECK-DAG:   %1 = alloca double, align 8
; CHECK-DAG:   store double %x, double* %1, align 8
; CHECK-DAG:   %2 = bitcast double* %1 to float*
; CHECK-DAG:   %3 = load float, float* %2, align 4
; CHECK-DAG:   store double %y, double* %1, align 8
; CHECK-DAG:   %4 = bitcast double* %1 to float*
; CHECK-DAG:   %5 = load float, float* %4, align 4
; CHECK-DAG:   %res = fcmp olt float %3, %5
; CHECK-DAG:   ret i1 %res

; CHECK: define internal i1 @__enzyme_done_truncate_op_func_64_52_32_23_f(double %x, double %y) {
; CHECK-DAG:   %enzyme_trunc = fptrunc double %x to float
; CHECK-DAG:   %enzyme_trunc1 = fptrunc double %y to float
; CHECK-DAG:   %res = fcmp olt float %enzyme_trunc, %enzyme_trunc1
; CHECK-DAG:   ret i1 %res
