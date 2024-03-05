; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define void @f(double* %x) {
  %y = load double, double* %x
  %m = fmul double %y, %y
  store double %m, double* %x
  ret void
}

declare void (double*)* @__enzyme_truncate_mem_func(...)
declare void (double*)* @__enzyme_truncate_op_func(...)

define void @tester(double* %data) {
entry:
  %ptr = call void (double*)* (...) @__enzyme_truncate_mem_func(void (double*)* @f, i64 64, i64 32)
  call void %ptr(double* %data)
  ret void
}
define void @tester_op(double* %data) {
entry:
  %ptr = call void (double*)* (...) @__enzyme_truncate_op_func(void (double*)* @f, i64 64, i64 32)
  call void %ptr(double* %data)
  ret void
}
define void @tester_op_mpfr(double* %data) {
entry:
  %ptr = call void (double*)* (...) @__enzyme_truncate_op_func(void (double*)* @f, i64 64, i64 3, i64 7)
  call void %ptr(double* %data)
  ret void
}

; CHECK: define internal void @__enzyme_done_truncate_mem_func_64_52to32_23_f(double* %x)
; CHECK-DAG:   %1 = alloca double, align 8
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   store double %y, double* %1, align 8
; CHECK-DAG:   %2 = bitcast double* %1 to float*
; CHECK-DAG:   %3 = load float, float* %2, align 4
; CHECK-DAG:   store double %y, double* %1, align 8
; CHECK-DAG:   %4 = bitcast double* %1 to float*
; CHECK-DAG:   %5 = load float, float* %4, align 4
; CHECK-DAG:   %m = fmul float %3, %5
; CHECK-DAG:   %6 = bitcast double* %1 to i64*
; CHECK-DAG:   store i64 0, i64* %6, align 4
; CHECK-DAG:   %7 = bitcast double* %1 to float*
; CHECK-DAG:   store float %m, float* %7, align 4
; CHECK-DAG:   %8 = load double, double* %1, align 8
; CHECK-DAG:   store double %8, double* %x, align 8
; CHECK-DAG:   ret void

; CHECK: define internal void @__enzyme_done_truncate_op_func_64_52to32_23_f(double* %x) {
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   %enzyme_trunc = fptrunc double %y to float
; CHECK-DAG:   %enzyme_trunc1 = fptrunc double %y to float
; CHECK-DAG:   %m = fmul float %enzyme_trunc, %enzyme_trunc1
; CHECK-DAG:   %enzyme_exp = fpext float %m to double
; CHECK-DAG:   store double %enzyme_exp, double* %x, align 8
; CHECK-DAG:   ret void

; CHECK: define internal void @__enzyme_done_truncate_op_func_64_52to11_7_f(double* %x) {
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   %m = call double @__enzyme_mpfr_64_52_binop_fmul(double %y, double %y, i64 3, i64 7)
; CHECK-DAG:   store double %m, double* %x, align 8
; CHECK-DAG:   ret void
