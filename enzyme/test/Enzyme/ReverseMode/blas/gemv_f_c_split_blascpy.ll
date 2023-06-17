;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-blas-copy=1 -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-blas-copy=1  -enzyme-lapack-copy=1 -S | FileCheck %s

;                       trans,                  M,                       N,                     alpha,                  A,    lda,                    x,  , incx,                  beta,                    y,  incy
declare void @dgemv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly) 

define void @f(i8* noalias %y, i8* noalias %A, i8* noalias %x) {
entry:
  %transa = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %alpha = alloca double, align 16
  %alpha_p = bitcast double* %alpha to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %incx = alloca i64, align 16
  %incx_p = bitcast i64* %incx to i8*
  %beta = alloca double, align 16
  %beta_p = bitcast double* %beta to i8*
  %incy = alloca i64, align 16
  %incy_p = bitcast i64* %incy to i8*
  store i8 78, i8* %transa, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store i64 2, i64* %incx, align 16
  store double 0.000000e+00, double* %beta
  store i64 1, i64* %incy, align 16
  call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta_p, i8* %y, i8* %incy_p) 
  ret void
}

define void @g(i8* noalias %y, i8* noalias %A, i8* noalias %x) {
entry:
  call void @f(i8* %y, i8* %A, i8* %x)
  %ptr = bitcast i8* %x to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %y, i8* %dy, i8* %A, i8* %dA, i8* %x, i8* %dx) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*)* @g, metadata !"enzyme_dup", i8* %y, i8* %dy, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %x, i8* %dx)
  ret void
}

; CHECK: define internal double* @augmented_f(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca double
; CHECK-NEXT:   %byref. = alloca i64
; CHECK-NEXT:   %incy = alloca i64, i64 1, align 16
; CHECK-NEXT:   %1 = bitcast i64* %incy to i8*
; CHECK-NEXT:   %beta = alloca double, i64 1, align 16
; CHECK-NEXT:   %2 = bitcast double* %beta to i8*
; CHECK-NEXT:   %incx = alloca i64, i64 1, align 16
; CHECK-NEXT:   %3 = bitcast i64* %incx to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %4 = bitcast i64* %lda to i8*
; CHECK-NEXT:   %alpha = alloca double, i64 1, align 16
; CHECK-NEXT:   %5 = bitcast double* %alpha to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %6 = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %7 = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %8 = bitcast i8* %7 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %6 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %5 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %alpha_p = bitcast double* %10 to i8*
; CHECK-NEXT:   %11 = bitcast i8* %4 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %11 to i8*
; CHECK-NEXT:   %12 = bitcast i8* %3 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incx_p = bitcast i64* %12 to i8*
; CHECK-NEXT:   %13 = bitcast i8* %2 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %beta_p = bitcast double* %13 to i8*
; CHECK-NEXT:   %14 = bitcast i8* %1 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incy_p = bitcast i64* %14 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 4, i64* %9, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %10, align 16
; CHECK-NEXT:   store i64 4, i64* %11, align 16
; CHECK-NEXT:   store i64 2, i64* %12, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %13
; CHECK-NEXT:   store i64 1, i64* %14, align 16
; CHECK-NEXT:   %trans_check = load i8, i8* %malloccall
; CHECK-NEXT:   %15 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %16 = load i64, i64* %15
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %16, 8
; CHECK-NEXT:   %malloccall8 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall8 to double*
; CHECK-NEXT:   store double* %cache.x, double** %0
; CHECK-NEXT:   store i64 1, i64* %byref.
; CHECK-NEXT:   call void @dcopy_64_(i8* %n_p, i8* %x, i8* %incx_p, double* %cache.x, i64* %byref.)
; CHECK-NEXT:   call void @dgemv_64_(i8* %malloccall, i8* %m_p, i8* %n_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta_p, i8* %y, i8* %incy_p)
; CHECK-NEXT:   %17 = load double*, double** %0
; CHECK-NEXT:   ret double* %17
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* %"x'", double* 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.incx = alloca i64
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %incy = alloca i64, i64 1, align 16
; CHECK-NEXT:   %1 = bitcast i64* %incy to i8*
; CHECK-NEXT:   %beta = alloca double, i64 1, align 16
; CHECK-NEXT:   %2 = bitcast double* %beta to i8*
; CHECK-NEXT:   %incx = alloca i64, i64 1, align 16
; CHECK-NEXT:   %3 = bitcast i64* %incx to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %4 = bitcast i64* %lda to i8*
; CHECK-NEXT:   %alpha = alloca double, i64 1, align 16
; CHECK-NEXT:   %5 = bitcast double* %alpha to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %6 = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %7 = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %8 = bitcast i8* %7 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %6 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %5 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %alpha_p = bitcast double* %10 to i8*
; CHECK-NEXT:   %11 = bitcast i8* %4 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %11 to i8*
; CHECK-NEXT:   %12 = bitcast i8* %3 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incx_p = bitcast i64* %12 to i8*
; CHECK-NEXT:   %13 = bitcast i8* %2 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %beta_p = bitcast double* %13 to i8*
; CHECK-NEXT:   %14 = bitcast i8* %1 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incy_p = bitcast i64* %14 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 4, i64* %9, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %10, align 16
; CHECK-NEXT:   store i64 4, i64* %11, align 16
; CHECK-NEXT:   store i64 2, i64* %12, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %13
; CHECK-NEXT:   store i64 1, i64* %14, align 16
; CHECK-NEXT:   %trans_check = load i8, i8* %malloccall
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %15 = bitcast double* %0 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.incx
; CHECK-NEXT:   %cast.incx = bitcast i64* %byref.incx to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %malloccall
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 0
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %23, i8* %byref.transpose.transa
; CHECK-NEXT:   call void @dger_64_(i8* %m_p, i8* %n_p, i8* %alpha_p, i8* %"y'", i8* %incy_p, i8* %15, i8* %cast.incx, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   call void bitcast (void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*)* @dgemv_64_ to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, double*, i8*, i8*)*)(i8* %byref.transpose.transa, i8* %m_p, i8* %n_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %"y'", i8* %incy_p, double* %byref.constant.fp.1.0, i8* %"x'", i8* %cast.incx)
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %byref.transpose.transa
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-DAG:    %[[r3:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %26 = or i1 %[[r3]], %[[r2]]
; CHECK-NEXT:   %27 = select i1 %26, i8* %m_p, i8* %n_p
; CHECK-NEXT:   call void @dscal_64_(i8* %27, i8* %beta_p, i8* %"y'", i8* %incy_p)
; CHECK-NEXT:   %28 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %28)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
