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
; CHECK-NEXT:   %byref.constant.char.r = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.char.l = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1 = alloca double, align 8
; CHECK-NEXT:   %byref.for.i = alloca i64, align 8
; CHECK-NEXT:   %byref.FMul = alloca double, align 8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.03 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.1 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.07 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.010 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.113 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.016 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.018 = alloca i64, align 8
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
; CHECK-NEXT:   store i8 114, i8* %byref.constant.char.r, align 1
; CHECK-NEXT:   store i8 108, i8* %byref.constant.char.l, align 1
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %trans, align 1
; CHECK-NEXT:   %1 = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %2 = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %3 = or i1 %2, %1
; CHECK-NEXT:   %4 = select i1 %3, i8* %byref.constant.char.l, i8* %byref.constant.char.r
; CHECK-NEXT:   %ld.row.trans1 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %5 = icmp eq i8 %ld.row.trans1, 110
; CHECK-NEXT:   %6 = icmp eq i8 %ld.row.trans1, 78
; CHECK-NEXT:   %7 = or i1 %6, %5
; CHECK-NEXT:   %8 = select i1 %7, i8* %n_p, i8* %k_p
; CHECK-NEXT:   %9 = select i1 %7, i8* %k_p, i8* %n_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1 = bitcast double* %byref.constant.fp.1 to i8*
; CHECK-NEXT:   call void @dsymm_64_(i8* %4, i8* %uplo, i8* %8, i8* %9, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %A, i8* %lda_p, i8* %fpcast.constant.fp.1, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %10 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %11 = load i64, i64* %10, align 4
; CHECK-NEXT:   %12 = icmp eq i64 %11, 0
; CHECK-NEXT:   br i1 %12, label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_loop:                                 ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   %13 = phi i64 [ 0, %invertentry ], [ %14, %invertentry_loop ]
; CHECK-NEXT:   %14 = add nuw nsw i64 %13, 1
; CHECK-NEXT:   store i64 %13, i64* %byref.for.i, align 4
; CHECK-NEXT:   %intcast.for.i = bitcast i64* %byref.for.i to i8*
; CHECK-NEXT:   %15 = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %16 = bitcast i8* %ldc_p to i64*
; CHECK-NEXT:   %17 = load i64, i64* %16, align 4
; CHECK-NEXT:   %18 = load i8, i8* %uplo, align 1
; CHECK-NEXT:   %19 = icmp eq i8 %18, 101
; CHECK-NEXT:   %20 = select i1 %19, i64 %17, i64 1
; CHECK-NEXT:   %21 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %22 = load i64, i64* %21, align 4
; CHECK-NEXT:   %23 = mul i64 %22, %20
; CHECK-NEXT:   %24 = select i1 %19, i64 1, i64 %17
; CHECK-NEXT:   %25 = bitcast i8* %intcast.for.i to i64*
; CHECK-NEXT:   %26 = load i64, i64* %25, align 4
; CHECK-NEXT:   %27 = mul i64 %26, %24
; CHECK-NEXT:   %28 = add i64 %23, %27
; CHECK-NEXT:   %29 = getelementptr double, double* %15, i64 %28
; CHECK-NEXT:   %30 = load double, double* %29, align 8
; CHECK-NEXT:   %31 = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %32 = load double, double* %31, align 8
; CHECK-NEXT:   %33 = fmul fast double %32, %30
; CHECK-NEXT:   store double %33, double* %byref.FMul, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   %intcast.constant.int.0 = bitcast i64* %byref.constant.int.0 to i8*
; CHECK-NEXT:   %ld.row.trans2 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %34 = icmp eq i8 %ld.row.trans2, 110
; CHECK-NEXT:   %35 = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %36 = or i1 %35, %34
; CHECK-NEXT:   %37 = select i1 %36, i8* %intcast.for.i, i8* %intcast.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.03, align 4
; CHECK-NEXT:   %intcast.constant.int.04 = bitcast i64* %byref.constant.int.03 to i8*
; CHECK-NEXT:   %ld.row.trans5 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %38 = icmp eq i8 %ld.row.trans5, 110
; CHECK-NEXT:   %39 = icmp eq i8 %ld.row.trans5, 78
; CHECK-NEXT:   %40 = or i1 %39, %38
; CHECK-NEXT:   %41 = select i1 %40, i8* %intcast.constant.int.04, i8* %intcast.for.i
; CHECK-NEXT:   %42 = bitcast i8* %A to double*
; CHECK-NEXT:   %43 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %44 = load i64, i64* %43, align 4
; CHECK-NEXT:   %45 = load i8, i8* %uplo, align 1
; CHECK-NEXT:   %46 = icmp eq i8 %45, 101
; CHECK-NEXT:   %47 = select i1 %46, i64 %44, i64 1
; CHECK-NEXT:   %48 = bitcast i8* %37 to i64*
; CHECK-NEXT:   %49 = load i64, i64* %48, align 4
; CHECK-NEXT:   %50 = mul i64 %49, %47
; CHECK-NEXT:   %51 = select i1 %46, i64 1, i64 %44
; CHECK-NEXT:   %52 = bitcast i8* %41 to i64*
; CHECK-NEXT:   %53 = load i64, i64* %52, align 4
; CHECK-NEXT:   %54 = mul i64 %53, %51
; CHECK-NEXT:   %55 = add i64 %50, %54
; CHECK-NEXT:   %56 = getelementptr double, double* %42, i64 %55
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1, align 4
; CHECK-NEXT:   %intcast.constant.int.1 = bitcast i64* %byref.constant.int.1 to i8*
; CHECK-NEXT:   %ld.row.trans6 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %57 = icmp eq i8 %ld.row.trans6, 110
; CHECK-NEXT:   %58 = icmp eq i8 %ld.row.trans6, 78
; CHECK-NEXT:   %59 = or i1 %58, %57
; CHECK-NEXT:   %60 = select i1 %59, i8* %lda_p, i8* %intcast.constant.int.1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.07, align 4
; CHECK-NEXT:   %intcast.constant.int.08 = bitcast i64* %byref.constant.int.07 to i8*
; CHECK-NEXT:   %ld.row.trans9 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %61 = icmp eq i8 %ld.row.trans9, 110
; CHECK-NEXT:   %62 = icmp eq i8 %ld.row.trans9, 78
; CHECK-NEXT:   %63 = or i1 %62, %61
; CHECK-NEXT:   %64 = select i1 %63, i8* %intcast.for.i, i8* %intcast.constant.int.08
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.010, align 4
; CHECK-NEXT:   %intcast.constant.int.011 = bitcast i64* %byref.constant.int.010 to i8*
; CHECK-NEXT:   %ld.row.trans12 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %65 = icmp eq i8 %ld.row.trans12, 110
; CHECK-NEXT:   %66 = icmp eq i8 %ld.row.trans12, 78
; CHECK-NEXT:   %67 = or i1 %66, %65
; CHECK-NEXT:   %68 = select i1 %67, i8* %intcast.constant.int.011, i8* %intcast.for.i
; CHECK-NEXT:   %69 = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %70 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %71 = load i64, i64* %70, align 4
; CHECK-NEXT:   %72 = load i8, i8* %uplo, align 1
; CHECK-NEXT:   %73 = icmp eq i8 %72, 101
; CHECK-NEXT:   %74 = select i1 %73, i64 %71, i64 1
; CHECK-NEXT:   %75 = bitcast i8* %64 to i64*
; CHECK-NEXT:   %76 = load i64, i64* %75, align 4
; CHECK-NEXT:   %77 = mul i64 %76, %74
; CHECK-NEXT:   %78 = select i1 %73, i64 1, i64 %71
; CHECK-NEXT:   %79 = bitcast i8* %68 to i64*
; CHECK-NEXT:   %80 = load i64, i64* %79, align 4
; CHECK-NEXT:   %81 = mul i64 %80, %78
; CHECK-NEXT:   %82 = add i64 %77, %81
; CHECK-NEXT:   %83 = getelementptr double, double* %69, i64 %82
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.113, align 4
; CHECK-NEXT:   %intcast.constant.int.114 = bitcast i64* %byref.constant.int.113 to i8*
; CHECK-NEXT:   %ld.row.trans15 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %84 = icmp eq i8 %ld.row.trans15, 110
; CHECK-NEXT:   %85 = icmp eq i8 %ld.row.trans15, 78
; CHECK-NEXT:   %86 = or i1 %85, %84
; CHECK-NEXT:   %87 = select i1 %86, i8* %lda_p, i8* %intcast.constant.int.114
; CHECK-NEXT:   call void @daxpy_64_(i8* %k_p, double* %byref.FMul, double* %56, i8* %60, double* %83, i8* %87)
; CHECK-NEXT:   %88 = icmp eq i64 %11, %14
; CHECK-NEXT:   br i1 %88, label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_end:                                  ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.016, align 4
; CHECK-NEXT:   %intcast.constant.int.017 = bitcast i64* %byref.constant.int.016 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.018, align 4
; CHECK-NEXT:   %intcast.constant.int.019 = bitcast i64* %byref.constant.int.018 to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %uplo, i8* %intcast.constant.int.017, i8* %intcast.constant.int.019, i8* %fpcast.constant.fp.1.0, i8* %beta_p, i8* %n_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %0, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
