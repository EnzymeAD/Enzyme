; RUN: if [ %llvmver -gt 12 ]; then if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi; fi
; RUN: if [ %llvmver -gt 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

define double @f(double %x) {
  %res = fadd double %x, 1.0
  ret double %res
}

declare double (double)* @__enzyme_truncate_mem_func(...)
declare double (double)* @__enzyme_truncate_op_func(...)

define double @tester(double %x) {
entry:
  %ptr = call double (double)* (...) @__enzyme_truncate_mem_func(double (double)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x)
  ret double %res
}
define double @tester_op_mpfr(double %x) {
entry:
  %ptr = call double (double)* (...) @__enzyme_truncate_op_func(double (double)* @f, i64 64, i64 3, i64 7)
  %res = call double %ptr(double %x)
  ret double %res
}

; CHECK: define internal double @__enzyme_done_truncate_mem_func_64_52to32_23_f(double %x) {
; CHECK-NEXT:   %1 = call double @__enzyme_fprt_64_52_const(double 1.000000e+00, i64 8, i64 23, i64 1)
; CHECK-NEXT:   %res = call double @__enzyme_fprt_64_52_binop_fadd(double %x, double %1, i64 8, i64 23, i64 1)
; CHECK-NEXT:   ret double %res
; CHECK-NEXT: }

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to11_7_f(double %x) {
; CHECK-NEXT:   %res = call double @__enzyme_fprt_64_52_binop_fadd(double %x, double 1.000000e+00, i64 3, i64 7, i64 2)
; CHECK-NEXT:   ret double %res
; CHECK-NEXT: }
