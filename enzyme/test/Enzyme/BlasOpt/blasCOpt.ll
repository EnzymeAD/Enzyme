;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -blas-opt -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="blas-opt" -S | FileCheck %s

declare void @dgemm_64_(i8, i8, i64, i64, i64, double, double* nocapture readonly, i64, double* nocapture readonly, i64, double, double* nocapture readonly, i64) 

declare void @dger_64_(i64 , i64, double, double* nocapture readonly, i64, double* nocapture readonly, i64, double*, i64) 

define void @f(double* noalias %C, double %alpha, double %beta, double* %x, double *%y, double* %v, double *%w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb) {
entry:
  %A = alloca double, align 16
  store double 0.000000e+00, double* %A
  %B = alloca double, align 16
  store double 0.000000e+00, double* %B
  call void @dger_64_(i64 %m, i64 %n, double %alpha, double* %x, i64 %incx, double* %y, i64 %incy, double* %A, i64 %lda) 
  call void @dger_64_(i64 %n, i64 %p, double %beta,  double* %v, i64 %incv, double* %w, i64 %incw, double* %B, i64 %ldb) 
  call void @dgemm_64_(i8 %transa, i8 %transb, i64 %m, i64 %n, i64 %p, double %alpha, double* %A, i64 %lda, double* %B, i64 %ldb, double %beta, double* %C, i64 %ldc) 
  ;%ptr = bitcast i8* %A to double*
  ;store double 0.0000000e+00, double* %ptr, align 8
  ret void
}


; CHECK: define void @f(double* noalias %C, double %alpha, double %beta, double* %x, double* %y, double* %v, double* %w, i64 %m, i64 %n, i64 %p, i64 %lda, i64 %ldb, i64 %ldc, i64 %incx, i64 %incy, i64 %incv, i64 %incw, i8 %transa, i8 %transb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %A = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %A
; CHECK-NEXT:   %B = alloca double, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %B
; CHECK-NEXT:   %0 = call double @ddot_64_(i64 %m, double* %y, i64 %incy, double* %v, i64 %incv) 
; CHECK-NEXT:   %1 = fmul double %alpha, %0
; CHECK-NEXT:   %2 = fmul double %1, %beta
; CHECK-NEXT:   call void @dger_64_(i64 %m, i64 %n, double %2, double* %x, i64 %incx, double* %w, i64 %incw, double* %C, i64 %ldc) 
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
