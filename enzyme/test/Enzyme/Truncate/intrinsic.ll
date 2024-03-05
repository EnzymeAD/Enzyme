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
; CHECK-DAG:   %1 = alloca double, align 8
; CHECK-DAG:   store double %x, double* %1, align 8
; CHECK-DAG:   %2 = bitcast double* %1 to float*
; CHECK-DAG:   %3 = load float, float* %2, align 4
; CHECK-DAG:   store double %y, double* %1, align 8
; CHECK-DAG:   %4 = bitcast double* %1 to float*
; CHECK-DAG:   %5 = load float, float* %4, align 4
; CHECK-DAG:   %res01 = call float @llvm.pow.f32(float %3, float %5)
; CHECK-DAG:   %6 = bitcast double* %1 to i64*
; CHECK-DAG:   store i64 0, i64* %6, align 4
; CHECK-DAG:   %7 = bitcast double* %1 to float*
; CHECK-DAG:   store float %res01, float* %7, align 4
; CHECK-DAG:   %8 = load double, double* %1, align 8
; CHECK-DAG:   store double %x, double* %1, align 8
; CHECK-DAG:   %9 = bitcast double* %1 to float*
; CHECK-DAG:   %10 = load float, float* %9, align 4
; CHECK-DAG:   store double %y, double* %1, align 8
; CHECK-DAG:   %11 = bitcast double* %1 to float*
; CHECK-DAG:   %12 = load float, float* %11, align 4
; CHECK-DAG:   %res12 = call float @llvm.pow.f32(float %10, float %12)
; CHECK-DAG:   %13 = bitcast double* %1 to i64*
; CHECK-DAG:   store i64 0, i64* %13, align 4
; CHECK-DAG:   %14 = bitcast double* %1 to float*
; CHECK-DAG:   store float %res12, float* %14, align 4
; CHECK-DAG:   %15 = load double, double* %1, align 8
; CHECK-DAG:   store double %x, double* %1, align 8
; CHECK-DAG:   %16 = bitcast double* %1 to float*
; CHECK-DAG:   %17 = load float, float* %16, align 4
; CHECK-DAG:   %res23 = call float @llvm.powi.f32.i16(float %17, i16 2)
; CHECK-DAG:   %18 = bitcast double* %1 to i64*
; CHECK-DAG:   store i64 0, i64* %18, align 4
; CHECK-DAG:   %19 = bitcast double* %1 to float*
; CHECK-DAG:   store float %res23, float* %19, align 4
; CHECK-DAG:   %20 = load double, double* %1, align 8
; CHECK-DAG:   store double %15, double* %1, align 8
; CHECK-DAG:   %21 = bitcast double* %1 to float*
; CHECK-DAG:   %22 = load float, float* %21, align 4
; CHECK-DAG:   store double %20, double* %1, align 8
; CHECK-DAG:   %23 = bitcast double* %1 to float*
; CHECK-DAG:   %24 = load float, float* %23, align 4
; CHECK-DAG:   %res = fadd float %22, %24
; CHECK-DAG:   %25 = bitcast double* %1 to i64*
; CHECK-DAG:   store i64 0, i64* %25, align 4
; CHECK-DAG:   %26 = bitcast double* %1 to float*
; CHECK-DAG:   store float %res, float* %26, align 4
; CHECK-DAG:   %27 = load double, double* %1, align 8
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %27

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to32_23_f(double %x, double %y) {
; CHECK-DAG:   %enzyme_trunc = fptrunc double %x to float
; CHECK-DAG:   %enzyme_trunc1 = fptrunc double %y to float
; CHECK-DAG:   %res02 = call float @llvm.pow.f32(float %enzyme_trunc, float %enzyme_trunc1)
; CHECK-DAG:   %enzyme_exp = fpext float %res02 to double
; CHECK-DAG:   %enzyme_trunc3 = fptrunc double %x to float
; CHECK-DAG:   %enzyme_trunc4 = fptrunc double %y to float
; CHECK-DAG:   %res15 = call float @llvm.pow.f32(float %enzyme_trunc3, float %enzyme_trunc4)
; CHECK-DAG:   %enzyme_exp6 = fpext float %res15 to double
; CHECK-DAG:   %enzyme_trunc7 = fptrunc double %x to float
; CHECK-DAG:   %res28 = call float @llvm.powi.f32.i16(float %enzyme_trunc7, i16 2)
; CHECK-DAG:   %enzyme_exp9 = fpext float %res28 to double
; CHECK-DAG:   %enzyme_trunc10 = fptrunc double %enzyme_exp6 to float
; CHECK-DAG:   %enzyme_trunc11 = fptrunc double %enzyme_exp9 to float
; CHECK-DAG:   %res = fadd float %enzyme_trunc10, %enzyme_trunc11
; CHECK-DAG:   %enzyme_exp12 = fpext float %res to double
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %enzyme_exp12

; CHECK: define internal double @__enzyme_done_truncate_op_func_64_52to11_7_f(double %x, double %y) {
; CHECK-DAG:   %1 = call double @__enzyme_mpfr_64_52_func_pow(double %x, double %y, i64 3, i64 7)
; CHECK-DAG:   %2 = call double @__enzyme_mpfr_64_52_intr_llvm_pow_f64(double %x, double %y, i64 3, i64 7)
; CHECK-DAG:   %3 = call double @__enzyme_mpfr_64_52_intr_llvm_powi_f64_i16(double %x, i16 2, i64 3, i64 7)
; CHECK-DAG:   %res = call double @__enzyme_mpfr_64_52_binop_fadd(double %2, double %3, i64 3, i64 7)
; CHECK-DAG:   call void @llvm.nvvm.barrier0()
; CHECK-DAG:   ret double %res
; CHECK-DAG: }
