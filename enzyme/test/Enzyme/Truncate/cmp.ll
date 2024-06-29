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
define i1 @tester_op_mpfr(double %x, double %y) {
entry:
  %ptr = call i1 (double, double)* (...) @__enzyme_truncate_op_func(i1 (double, double)* @f, i64 64, i64 3, i64 7)
  %res = call i1 %ptr(double %x, double %y)
  ret i1 %res
}

; CHECK: define internal i1 @__enzyme_done_truncate_mem_func_64_52to32_23_f(double %x, double %y) {
; CHECK-NEXT:   %res = call i1 @__enzyme_fprt_64_52_fcmp_olt(double %x, double %y, i64 8, i64 23, i64 1, {{.*}}i8{{.*}})
; CHECK-NEXT:   ret i1 %res
; CHECK-NEXT: }

; CHECK: define internal i1 @__enzyme_done_truncate_op_func_64_52to32_23_f(double %x, double %y) {
; CHECK-NEXT:   %res = fcmp olt double %x, %y
; CHECK-NEXT:   ret i1 %res
; CHECK-NEXT: }

; CHECK: define internal i1 @__enzyme_done_truncate_op_func_64_52to11_7_f(double %x, double %y) {
; CHECK-NEXT:   %res = fcmp olt double %x, %y
; CHECK-NEXT:   ret i1 %res
; CHECK-NEXT: }
