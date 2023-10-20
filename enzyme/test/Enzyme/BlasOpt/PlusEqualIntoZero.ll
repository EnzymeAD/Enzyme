;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -blas-opt -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="blas-opt" -S | FileCheck %s

declare void @dgemv_64_(i8, i64, i64, double, double* nocapture readonly, i64, double* nocapture readonly, i64, double, double* nocapture, i64)
declare void @dgemm_64_(i8, i8, i64, i64, i64, double, double* nocapture readonly, i64, double* nocapture readonly, i64, double, double* nocapture, i64) 
declare void @daxpy(i32, double, double* nocapture readonly, i32, double* nocapture, i32)

declare void @llvm.memset.p0.i64(double* nocapture writeonly, i8, i64, i1)
declare void @llvm.memset.inline.p0.i64(double* nocapture writeonly, i8, i64, i1)


; y is zeroed, so we can replace the beta arg with const 0.0
define void @h(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  %size = mul i64 %incy, %n
  call void @llvm.memset.p0.i64(double* %y, i8 0, i64 %size, i1 false) 
  call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double %beta, double* %y, i64 %incy)
  ret void
}

define void @h2(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  %size = mul i64 %incy, %n
  call void @llvm.memset.p0.i64(double* %y, i8 0, i64 %size, i1 false) 
  store double 1.000000e+00, double* %y
  call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double %beta, double* %y, i64 %incy)
  ret void
}

define void @h3(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  call void @llvm.memset.p0.i64(double* %y, i8 0, i64 2, i1 false) 
  call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double %beta, double* %y, i64 %incy)
  ret void
}

; x is zeroed, so y = alpha * x + y is a no-op, so we remove the call to daxpy
define void @g(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  %size = mul i64 %incx, %n
  call void @llvm.memset.p0.i64(double* %x, i8 0, i64 %size, i1 false) 
  call void @daxpy(i64 %n, double %alpha, double* %x, i64 %incx, double* %y, i64 %incy)
  ret void
}


; A is zeroed, so we can replace the beta arg with const 0.0
define void @f(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  %size = mul i64 %ldc, %n
  call void @llvm.memset.p0.i64(double* %C, i8 0, i64 %size, i1 false) 
  call void @dgemm_64_(i8 %transa, i8 %transb, i64 %m, i64 %n, i64 %p, double %alpha, double* %A, i64 %lda, double* %B, i64 %ldb, double %beta, double* %C, i64 %ldc) 
  ret void
}

; A is zeroed, so we can replace the beta arg with const 0.0
; use inline memset here to check we still recognize it
define void @f2(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  %size = mul i64 %ldc, %n
  call void @llvm.memset.inline.p0.i64(double* %C, i8 0, i64 %size, i1 false) 
  call void @dgemm_64_(i8 %transa, i8 %transb, i64 %m, i64 %n, i64 %p, double %alpha, double* %A, i64 %lda, double* %B, i64 %ldb, double %beta, double* %C, i64 %ldc) 
  ret void
}

; CHECK: define void @h(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double 0.0, double* %y, i64 %incy)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @h2(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double 0.0, double* %y, i64 %incy)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @h3(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   call void @dgemv_64_(i8 %transa, i64 %m, i64 %n, double %alpha, double* %A, i64 %lda, double* %x, i64 %incx, double 0.0, double* %y, i64 %incy)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @g(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   %size = mul i64 %incx, %n
; CHECK-NEXT:   call void @llvm.memset.p0.i64(double* %x, i8 0, i64 %size, i1 false) 
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @f(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   call void @dgemm_64_(i8 %transa, i8 %transb, i64 %m, i64 %n, i64 %p, double %alpha, double* %A, i64 %lda, double* %B, i64 %ldb, double 0.0, double* %C, i64 %ldc) 
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define void @f2(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   call void @dgemm_64_(i8 %transa, i8 %transb, i64 %m, i64 %n, i64 %p, double %alpha, double* %A, i64 %lda, double* %B, i64 %ldb, double 0.0, double* %C, i64 %ldc) 
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
