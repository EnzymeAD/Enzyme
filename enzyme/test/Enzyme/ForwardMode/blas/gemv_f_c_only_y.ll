;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-blas-copy=1 -enzyme-lapack-copy=1 -enzyme-detect-readthrow=0 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-blas-copy=1 -enzyme-lapack-copy=1 -enzyme-detect-readthrow=0 -S | FileCheck %s

; Only the output y is active. Its tangent must follow BLAS semantics for beta:
; beta == 0 overwrites y, so dy must be overwritten as well (not scaled by 0,
; which keeps NaN or Inf). The rule therefore calls gemv with alpha = 0 rather
; than scal(beta, dy).

;                       trans,                  M,                       N,                     alpha,                  A,    lda,                    x,  , incx,                  beta,                    y,  incy
declare void @dgemv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i64)

define void @f(i8* noalias %y, i8* noalias %A, i8* noalias %x, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %transa = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %incx = alloca i64, align 16
  %incx_p = bitcast i64* %incx to i8*
  %incy = alloca i64, align 16
  %incy_p = bitcast i64* %incy to i8*
  store i8 78, i8* %transa, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 4, i64* %lda, align 16
  store i64 2, i64* %incx, align 16
  store i64 1, i64* %incy, align 16
  call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p, i64 1)
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @active(i8* %y, i8* %dy, i8* %A, i8* %x, i8* %alpha, i8* %beta) {
entry:
  call void (...) @__enzyme_fwddiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %y, i8* %dy, metadata !"enzyme_const", i8* %A, metadata !"enzyme_const", i8* %x, metadata !"enzyme_const", i8* %alpha, metadata !"enzyme_const", i8* %beta)
  ret void
}

; CHECK: define internal void @fwddiffef(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* noalias %x, i8* noalias %alpha, i8* noalias %beta)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.N = alloca i8, align 1
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %incx = alloca i64, align 16
; CHECK-NEXT:   %incx_p = bitcast i64* %incx to i8*
; CHECK-NEXT:   %incy = alloca i64, align 16
; CHECK-NEXT:   %incy_p = bitcast i64* %incy to i8*
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 2, i64* %incx, align 16
; CHECK-NEXT:   store i64 1, i64* %incy, align 16
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 4
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   store i8 78, i8* %byref.constant.char.N, align 1
; CHECK-NEXT:   call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %fpcast.constant.fp.0.0, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %"y'", i8* %incy_p, i64 1)
; CHECK-NEXT:   call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
