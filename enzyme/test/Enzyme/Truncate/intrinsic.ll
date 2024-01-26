; RUN: if [ %llvmver -gt 12 ]; then if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi; fi
; RUN: if [ %llvmver -gt 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @llvm.pow.f64(double %Val, double %Power)
declare double @llvm.powi.f64.i16(double %Val, i16 %power)
declare void @llvm.nvvm.barrier0()

define double @f(double %x, double %y) {
  %res1 = call double @llvm.pow.f64(double %x, double %y)
  %res2 = call double @llvm.powi.f64.i16(double %x, i16 2)
  %res = fadd double %res1, %res2
  call void @llvm.nvvm.barrier0()
  ret double %res
}

declare double (double, double)* @__enzyme_truncate_func(...)

define double @tester(double %x, double %y) {
entry:
  %ptr = call double (double, double)* (...) @__enzyme_truncate_func(double (double, double)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y)
  ret double %res
}

; CHECK: define internal double @trunc_64_32f(double %x, double %y) {
; CHECK-NEXT:   %1 = alloca double, align 8
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   %2 = bitcast double* %1 to float*
; CHECK-NEXT:   %3 = load float, float* %2, align 4
; CHECK-NEXT:   store double %y, double* %1, align 8
; CHECK-NEXT:   %4 = bitcast double* %1 to float*
; CHECK-NEXT:   %5 = load float, float* %4, align 4
; CHECK-NEXT:   %res11 = call float @llvm.pow.f32(float %3, float %5)
; CHECK-NEXT:   %6 = bitcast double* %1 to i64*
; CHECK-NEXT:   store i64 0, i64* %6, align 4
; CHECK-NEXT:   %7 = bitcast double* %1 to float*
; CHECK-NEXT:   store float %res11, float* %7, align 4
; CHECK-NEXT:   %8 = load double, double* %1, align 8
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   %9 = bitcast double* %1 to float*
; CHECK-NEXT:   %10 = load float, float* %9, align 4
; CHECK-NEXT:   %res22 = call float @llvm.powi.f32.i16(float %10, i16 2)
; CHECK-NEXT:   %11 = bitcast double* %1 to i64*
; CHECK-NEXT:   store i64 0, i64* %11, align 4
; CHECK-NEXT:   %12 = bitcast double* %1 to float*
; CHECK-NEXT:   store float %res22, float* %12, align 4
; CHECK-NEXT:   %13 = load double, double* %1, align 8
; CHECK-NEXT:   store double %8, double* %1, align 8
; CHECK-NEXT:   %14 = bitcast double* %1 to float*
; CHECK-NEXT:   %15 = load float, float* %14, align 4
; CHECK-NEXT:   store double %13, double* %1, align 8
; CHECK-NEXT:   %16 = bitcast double* %1 to float*
; CHECK-NEXT:   %17 = load float, float* %16, align 4
; CHECK-NEXT:   %res = fadd float %15, %17
; CHECK-NEXT:   %18 = bitcast double* %1 to i64*
; CHECK-NEXT:   store i64 0, i64* %18, align 4
; CHECK-NEXT:   %19 = bitcast double* %1 to float*
; CHECK-NEXT:   store float %res, float* %19, align 4
; CHECK-NEXT:   %20 = load double, double* %1, align 8
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   ret double %20
; CHECK-NEXT: }
