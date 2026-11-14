; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s

; Verify forward-mode BLAS matrix derivatives (gemm, symm, syrk, cblas_*, and
; Julia integer-pointer ABI dgemm_64_) with active beta and C emit the
; matrix-aware AxpyMat helper after beta-scaling dC instead of failing on
; AssertingInactiveArg or mis-sizing a 1D axpy call.

declare void @__enzyme_fwddiff(ptr, ...)
declare void @dgemm_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64)
declare void @dsymm_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64)
declare void @dsyrk_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64)
declare void @dgemm_64_(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define void @f_dgemm(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc) {
  call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)
  ret void
}

define void @f_dsymm(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc) {
  call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)
  ret void
}

define void @f_dsyrk(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %beta, ptr %C, ptr %ldc) {
  call void @dsyrk_(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)
  ret void
}

define void @f_julia_gemm(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc) {
  %i0 = ptrtoint ptr %ta to i64
  %i1 = ptrtoint ptr %tb to i64
  %i2 = ptrtoint ptr %m to i64
  %i3 = ptrtoint ptr %n to i64
  %i4 = ptrtoint ptr %k to i64
  %i5 = ptrtoint ptr %alpha to i64
  %i6 = ptrtoint ptr %A to i64
  %i7 = ptrtoint ptr %lda to i64
  %i8 = ptrtoint ptr %B to i64
  %i9 = ptrtoint ptr %ldb to i64
  %i10 = ptrtoint ptr %beta to i64
  %i11 = ptrtoint ptr %C to i64
  %i12 = ptrtoint ptr %ldc to i64
  call void @dgemm_64_(i64 %i0, i64 %i1, i64 %i2, i64 %i3, i64 %i4, i64 %i5, i64 %i6, i64 %i7, i64 %i8, i64 %i9, i64 %i10, i64 %i11, i64 %i12, i64 1, i64 1)
  ret void
}

define void @test_all(ptr %side, ptr %uplo, ptr %ta, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %dA, ptr %lda, ptr %B, ptr %dB, ptr %ldb, ptr %beta, ptr %dbeta, ptr %C, ptr %dC, ptr %ldc) {
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_dgemm,
    metadata !"enzyme_const", ptr %ta, metadata !"enzyme_const", ptr %ta,
    metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %n, metadata !"enzyme_const", ptr %k,
    metadata !"enzyme_const", ptr %alpha,
    metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %lda,
    metadata !"enzyme_dup", ptr %B, ptr %dB, metadata !"enzyme_const", ptr %ldb,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", ptr %ldc)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_dsymm,
    metadata !"enzyme_const", ptr %side, metadata !"enzyme_const", ptr %uplo,
    metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %n,
    metadata !"enzyme_const", ptr %alpha,
    metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %lda,
    metadata !"enzyme_dup", ptr %B, ptr %dB, metadata !"enzyme_const", ptr %ldb,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", ptr %ldc)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_dsyrk,
    metadata !"enzyme_const", ptr %uplo, metadata !"enzyme_const", ptr %ta,
    metadata !"enzyme_const", ptr %n, metadata !"enzyme_const", ptr %k,
    metadata !"enzyme_const", ptr %alpha,
    metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %lda,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", ptr %ldc)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_julia_gemm,
    metadata !"enzyme_const", ptr %ta, metadata !"enzyme_const", ptr %ta,
    metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %n, metadata !"enzyme_const", ptr %k,
    metadata !"enzyme_const", ptr %alpha,
    metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %lda,
    metadata !"enzyme_dup", ptr %B, ptr %dB, metadata !"enzyme_const", ptr %ldb,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", ptr %ldc)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffef_dgemm(
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %"B'", ptr %ldb, ptr %beta, ptr %"C'", ptr %ldc, i64 1, i64 1)
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %"A'", ptr %lda, ptr %B, ptr %ldb, ptr %byref.constant.fp.1{{.*}}, ptr %"C'", ptr %ldc, i64 1, i64 1)
; CHECK: call void @daxpy_(
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)

; CHECK-LABEL: define internal void @__enzyme_axpy_mat_daxpy__32_ap(

; CHECK-LABEL: define internal void @fwddiffef_dsymm(
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %"B'", ptr %ldb, ptr %beta, ptr %"C'", ptr %ldc, i64 1, i64 1)
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %"A'", ptr %lda, ptr %B, ptr %ldb, ptr %byref.constant.fp.1{{.*}}, ptr %"C'", ptr %ldc, i64 1, i64 1)
; CHECK: call void @daxpy_(
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)

; CHECK-LABEL: define internal void @fwddiffef_dsyrk(
; CHECK: call void @dsyr2k_(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %"A'", ptr %lda, ptr %beta, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @daxpy_(
; CHECK: call void @dsyrk_(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)

; CHECK-LABEL: define internal void @fwddiffef_julia_gemm(
; CHECK: call void @dgemm_64_(
; CHECK: call void @dgemm_64_(
; CHECK: call void @daxpy_64_(
; CHECK: call void @dgemm_64_(

; CHECK-LABEL: define internal void @__enzyme_axpy_mat_daxpy_64__64_jl_ai64(
