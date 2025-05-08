; RUN: if [ %llvmver -gt 12 ]; then if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi; fi
; RUN: if [ %llvmver -gt 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @pow(double %Val, double %Power)
declare double @llvm.pow.f64(double %Val, double %Power)
declare double @llvm.powi.f64.i16(double %Val, i16 %power)
declare void @llvm.nvvm.barrier0()

define double @f(double %x, double %y) {
  %res0 = call double @pow(double %x, double %y)
  %res1 = call double @llvm.pow.f64(double %x, double %y)
  %res2 = call double @llvm.powi.f64.i16(double %x, i16 2)
  %res = fadd double %res1, %res2
  call void @llvm.nvvm.barrier0()
  ret double %res
}

declare double (double, double)* @__enzyme_truncate_mem_func(...)
declare double (double, double)* @__enzyme_truncate_op_func(...)

define double @tester(double %x, double %y) {
entry:
  %ptr = call double (double, double)* (...) @__enzyme_truncate_mem_func(double (double, double)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y)
  ret double %res
}
; TODO This used to test if we detect that we truncate to a native float type
; and use that instead of MPFR but now we always generate the FPRT calls.
; Instead we shuold probably add an additional flag/mode to truncate to native
; types
define double @tester_op(double %x, double %y) {
entry:
  %ptr = call double (double, double)* (...) @__enzyme_truncate_op_func(double (double, double)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y)
  ret double %res
}
define double @tester_op_mpfr(double %x, double %y) {
entry:
  %ptr = call double (double, double)* (...) @__enzyme_truncate_op_func(double (double, double)* @f, i64 64, i64 3, i64 7)
  %res = call double %ptr(double %x, double %y)
  ret double %res
}

; CHECK: define internal double @__enzyme_done_truncate_mem_func_64_52to32_23_f(double %x, double %y) {
; CHECK-DAG:   %1 = call double @__enzyme_fprt_64_52_func_pow(double %x, double %y, i64 8, i64 23, i64 1, {{.*}}i8{{.*}})
; CHECK-DAG:   %2 = call double @__enzyme_fprt_64_52_intr_llvm_pow_f64(double %x, double %y, i64 8, i64 23, i64 1, {{.*}}i8{{.*}})
; CHECK-DAG:   %3 = call double @__enzyme_fprt_64_52_intr_llvm_powi_f64_i16(double %x, i16 2, i64 8, i64 23, i64 1, {{.*}}i8{{.*}})
; CHECK-DAG:   %res = call double @__enzyme_fprt_64_52_binop_fadd(double %2, double %3, i64 8, i64 23, i64 1, {{.*}}i8{{.*}})
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %res
; CHECK-DAG: }

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to32_23_f(double %x, double %y) {
; CHECK-DAG:   %1 = call double @__enzyme_fprt_64_52_func_pow(double %x, double %y, i64 8, i64 23, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %2 = call double @__enzyme_fprt_64_52_intr_llvm_pow_f64(double %x, double %y, i64 8, i64 23, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %3 = call double @__enzyme_fprt_64_52_intr_llvm_powi_f64_i16(double %x, i16 2, i64 8, i64 23, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %res = call double @__enzyme_fprt_64_52_binop_fadd(double %2, double %3, i64 8, i64 23, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %res
; CHECK-DAG: }

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to11_7_f(double %x, double %y) {
; CHECK-DAG:   %1 = call double @__enzyme_fprt_64_52_func_pow(double %x, double %y, i64 3, i64 7, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %2 = call double @__enzyme_fprt_64_52_intr_llvm_pow_f64(double %x, double %y, i64 3, i64 7, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %3 = call double @__enzyme_fprt_64_52_intr_llvm_powi_f64_i16(double %x, i16 2, i64 3, i64 7, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   %res = call double @__enzyme_fprt_64_52_binop_fadd(double %2, double %3, i64 3, i64 7, i64 2, {{.*}}i8{{.*}})
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %res
; CHECK-DAG: }
