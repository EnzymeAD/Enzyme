;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S -enzyme-detect-readthrow=0 | FileCheck %s

; Only the output C is active. Its tangent must follow BLAS semantics for beta:
; beta == 0 overwrites C, so dC must be overwritten as well (not scaled by 0,
; which keeps NaN or Inf). The rule therefore calls gemm with alpha = 0 rather
; than lascl(beta, dC).

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i64, i64)

define void @f(i8* noalias %C, i8* noalias %A, i8* noalias %B, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %ldb = alloca i64, align 16
  %ldb_p = bitcast i64* %ldb to i8*
  %ldc = alloca i64, align 16
  %ldc_p = bitcast i64* %ldc to i8*
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p, i64 1, i64 1)
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %B, i8* %alpha, i8* %beta) {
entry:
  call void (...) @__enzyme_fwddiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_const", i8* %A, metadata !"enzyme_const", i8* %B, metadata !"enzyme_const", i8* %alpha, metadata !"enzyme_const", i8* %beta)
  ret void
}

; CHECK: define internal void @fwddiffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* noalias %B, i8* noalias %alpha, i8* noalias %beta)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double, align 8
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %ldb = alloca i64, align 16
; CHECK-NEXT:   %ldb_p = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 4
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %fpcast.constant.fp.0.0, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %"C'", i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
