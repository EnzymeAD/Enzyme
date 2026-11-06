; RUN: %opt < %s %loadEnzyme %enzyme -S | FileCheck %s --implicit-check-not='call void @dgetrs_' --implicit-check-not='call void @dtrmm_' --implicit-check-not='call void @dgemm_' --implicit-check-not='call void @dlaswp_'
;
; Existing Fortran declarations may omit hidden CHARACTER lengths. Preserve
; their exact signatures and calling conventions in both derivative modes.
; Activating the supplied factors and RHS exercises GETRS, TRMM, GEMM and LASWP.

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@normal = private constant i8 78

declare fastcc void @dgetrs_(i8*, i32*, i32*, double*, i32*, i32*, double*, i32*, i32*)
declare fastcc void @dtrmm_(i8*, i8*, i8*, i8*, i32*, i32*, double*, double*, i32*, double*, i32*)
declare fastcc void @dgemm_(i8*, i8*, i32*, i32*, i32*, double*, double*, i32*, double*, i32*, double*, double*, i32*)
declare fastcc void @dlaswp_(i32*, double*, i32*, i32*, i32*, i32*, i32*)
declare void @__enzyme_fwddiff(...)
declare void @__enzyme_autodiff(...)

; Exclude commas and closing parentheses from each argument so an extra hidden
; length cannot be consumed by a wildcard. Pointer spelling and attributes may
; differ between typed-pointer and opaque-pointer LLVM versions.
; CHECK-DAG: declare fastcc void @dgetrs_({{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}})
; CHECK-DAG: declare fastcc void @dtrmm_({{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}})
; CHECK-DAG: declare fastcc void @dgemm_({{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}})
; CHECK-DAG: declare fastcc void @dlaswp_({{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}}, {{[^,)]*}})
; CHECK-DAG: call fastcc void @dgetrs_(
; CHECK-DAG: call fastcc void @dtrmm_(
; CHECK-DAG: call fastcc void @dgemm_(
; CHECK-DAG: call fastcc void @dlaswp_(

define void @solve(double* noalias %lu, i32* noalias %piv, double* noalias %rhs) {
entry:
  %n = alloca i32, align 4
  %nrhs = alloca i32, align 4
  %info = alloca i32, align 4
  store i32 2, i32* %n, align 4
  store i32 1, i32* %nrhs, align 4
  store i32 0, i32* %info, align 4
  call fastcc void @dgetrs_(i8* @normal, i32* %n, i32* %nrhs, double* %lu,
                            i32* %n, i32* %piv, double* %rhs, i32* %n, i32* %info)
  ret void
}

define void @forward(double* %lu, double* %dlu, i32* %piv, double* %rhs, double* %drhs) {
entry:
  call void (...) @__enzyme_fwddiff(void (double*, i32*, double*)* @solve,
      metadata !"enzyme_dup", double* %lu, double* %dlu,
      metadata !"enzyme_const", i32* %piv,
      metadata !"enzyme_dup", double* %rhs, double* %drhs)
  ret void
}

define void @reverse(double* %lu, double* %dlu, i32* %piv, double* %rhs, double* %drhs) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, i32*, double*)* @solve,
      metadata !"enzyme_dup", double* %lu, double* %dlu,
      metadata !"enzyme_const", i32* %piv,
      metadata !"enzyme_dup", double* %rhs, double* %drhs)
  ret void
}
