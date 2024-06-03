;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

; dsyrk	(	character 	UPLO,
; character 	TRANS,
; integer 	N,
; integer 	K,
; double precision 	ALPHA,
; double precision, dimension(lda,*) 	A,
; integer 	LDA,
; double precision 	BETA,
; double precision, dimension(ldc,*) 	C,
; integer 	LDC
; )

declare void @dsyrk_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture, i8* nocapture readonly, i64, i64) 

define void @f(i8* %C, i8* %A) {
entry:
  %uplo = alloca i8, align 1
  %trans = alloca i8, align 1
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
  %alpha = alloca double, align 16
  %alpha_p = bitcast double* %alpha to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %ldb = alloca i64, align 16
  %ldb_p = bitcast i64* %ldb to i8*
  %beta = alloca double, align 16
  %beta_p = bitcast double* %beta to i8*
  %ldc = alloca i64, align 16
  %ldc_p = bitcast i64* %ldc to i8*
  store i8 85, i8* %uplo, align 1
  store i8 78, i8* %trans, align 1
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  call void @dsyrk_64_(i8* %uplo, i8* %trans, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1) 
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA)
  ret void
}

; CHECK: define internal void @diffef(i8* %C, i8* %"C'", i8* %A, i8* %"A'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   %byref.trans_to_side.uplo = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1 = alloca double, align 8
; CHECK-NEXT:   %byref.for.i = alloca i64, align 8
; CHECK-NEXT:   %byref.mul = alloca double, align 8
; CHECK-NEXT:   %byref.constant.int.1 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.11 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.03 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %0 = alloca i64, align 8
; CHECK-NEXT:   %uplo = alloca i8, align 1
; CHECK-NEXT:   %trans = alloca i8, align 1
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %alpha = alloca double, align 16
; CHECK-NEXT:   %alpha_p = bitcast double* %alpha to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %beta = alloca double, align 16
; CHECK-NEXT:   %beta_p = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   store i8 85, i8* %uplo, align 1
; CHECK-NEXT:   store i8 78, i8* %trans, align 1
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta, align 8
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   call void @dsyrk_64_(i8* %uplo, i8* %trans, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 4
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   %ld.uplo = load i8, i8* %uplo, align 1
; CHECK-NEXT:   %1 = icmp eq i8 %ld.uplo, 78
; CHECK-NEXT:   %2 = select i1 %1, i8 76, i8 108
; CHECK-NEXT:   %3 = icmp eq i8 %ld.uplo, 116
; CHECK-NEXT:   %4 = select i1 %3, i8 114, i8 %2
; CHECK-NEXT:   %5 = icmp eq i8 %ld.uplo, 84
; CHECK-NEXT:   %6 = select i1 %5, i8 82, i8 %4
; CHECK-NEXT:   store i8 %6, i8* %byref.trans_to_side.uplo, align 1
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %trans, align 1
; CHECK-NEXT:   %7 = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %8 = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %9 = or i1 %8, %7
; CHECK-NEXT:   %10 = select i1 %9, i8* %n_p, i8* %k_p
; CHECK-NEXT:   %11 = select i1 %9, i8* %k_p, i8* %n_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1 = bitcast double* %byref.constant.fp.1 to i8*
; CHECK-NEXT:   call void @dsymm_64_(i8* %byref.trans_to_side.uplo, i8* %uplo, i8* %10, i8* %11, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %A, i8* %lda_p, i8* %fpcast.constant.fp.1, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %12 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %13 = load i64, i64* %12, align 4
; CHECK-NEXT:   %14 = icmp eq i64 %13, 0
; CHECK-NEXT:   br i1 %14, label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_loop:                                 ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   %15 = phi i64 [ 0, %invertentry ], [ %44, %invertentry_loop ]
; CHECK-NEXT:   store i64 %15, i64* %byref.for.i, align 4
; CHECK-NEXT:   %intcast.for.i = bitcast i64* %byref.for.i to i8*
; CHECK-NEXT:   %16 = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %17 = bitcast i8* %ldc_p to i64*
; CHECK-NEXT:   %18 = load i64, i64* %17, align 4
; CHECK-NEXT:   %19 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %20 = load i64, i64* %19, align 4
; CHECK-NEXT:   %21 = mul i64 %20, %18
; CHECK-NEXT:   %22 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %23 = load i64, i64* %22, align 4
; CHECK-NEXT:   %24 = add i64 %21, %23
; CHECK-NEXT:   %25 = getelementptr double, double* %16, i64 %24
; CHECK-NEXT:   %26 = load double, double* %25, align 8
; CHECK-NEXT:   %27 = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %28 = load double, double* %27, align 8
; CHECK-NEXT:   %29 = fmul fast double %28, %26
; CHECK-NEXT:   store double %29, double* %byref.mul, align 8
; CHECK-NEXT:   %30 = bitcast i8* %A to double*
; CHECK-NEXT:   %31 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %32 = load i64, i64* %31, align 4
; CHECK-NEXT:   %33 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %34 = load i64, i64* %33, align 4
; CHECK-NEXT:   %35 = mul i64 %34, %32
; CHECK-NEXT:   %36 = getelementptr double, double* %30, i64 %35
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1, align 4
; CHECK-NEXT:   %intcast.constant.int.1 = bitcast i64* %byref.constant.int.1 to i8*
; CHECK-NEXT:   %37 = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %38 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %39 = load i64, i64* %38, align 4
; CHECK-NEXT:   %40 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %41 = load i64, i64* %40, align 4
; CHECK-NEXT:   %42 = mul i64 %41, %39
; CHECK-NEXT:   %43 = getelementptr double, double* %37, i64 %42
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.11, align 4
; CHECK-NEXT:   %intcast.constant.int.12 = bitcast i64* %byref.constant.int.11 to i8*
; CHECK-NEXT:   call void @daxpy_64_(i8* %k_p, double* %byref.mul, double* %36, i8* %intcast.constant.int.1, double* %43, i8* %intcast.constant.int.12)
; CHECK-NEXT:   %44 = add nuw nsw i64 %13, 1
; CHECK-NEXT:   %45 = icmp eq i64 %13, %44
; CHECK-NEXT:   br i1 %45, label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_end:                                  ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   %intcast.constant.int.0 = bitcast i64* %byref.constant.int.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.03, align 4
; CHECK-NEXT:   %intcast.constant.int.04 = bitcast i64* %byref.constant.int.03 to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %uplo, i8* %intcast.constant.int.0, i8* %intcast.constant.int.04, i8* %fpcast.constant.fp.1.0, i8* %beta_p, i8* %n_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %0, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
