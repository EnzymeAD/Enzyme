; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S -enzyme-detect-readthrow=0 | FileCheck %s

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
; TODO This used to test if we detect that we truncate to a native float type
; and use that instead of MPFR but now we always generate the FPRT calls.
; Instead we shuold probably add an additional flag/mode to truncate to native
; types
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

; CHECK: define internal void @__enzyme_done_truncate_mem_func_64_52to32_23_f(double* %x) {
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   %m = call double @__enzyme_fprt_64_52_binop_fmul(double %y, double %y, i64 8, i64 23, i64 1)
; CHECK-DAG:   store double %m, double* %x, align 8
; CHECK-DAG:   ret void
; CHECK-DAG: }

; CHECK: define internal void @__enzyme_done_truncate_op_func_64_52to32_23_f(double* %x) {
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   %m = call double @__enzyme_fprt_64_52_binop_fmul(double %y, double %y, i64 8, i64 23, i64 2)
; CHECK-DAG:   store double %m, double* %x, align 8
; CHECK-DAG:   ret void
; CHECK-DAG: }

; CHECK: define internal void @__enzyme_done_truncate_op_func_64_52to11_7_f(double* %x) {
; CHECK-DAG:   %y = load double, double* %x, align 8
; CHECK-DAG:   %m = call double @__enzyme_fprt_64_52_binop_fmul(double %y, double %y, i64 3, i64 7, i64 2)
; CHECK-DAG:   store double %m, double* %x, align 8
; CHECK-DAG:   ret void
; CHECK-DAG: }
