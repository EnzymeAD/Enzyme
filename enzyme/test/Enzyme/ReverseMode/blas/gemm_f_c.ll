;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly) 

define void @f(i8* %C, i8* %A, i8* %B) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
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
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* noundef nonnull %transa, i8* noundef nonnull %transb, i8* noundef nonnull %m_p, i8* noundef nonnull %n_p, i8* noundef nonnull %k_p, i8* noundef nonnull %alpha_p, i8* %A, i8* noundef nonnull %lda_p, i8* %B, i8* noundef nonnull %ldb_p, i8* noundef nonnull %beta_p, i8* %C, i8* noundef nonnull %ldc_p) 
  %ptr = bitcast i8* %A to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %B, i8* %dB)
  ret void
}

; CHECK: define internal void @diffef(i8* %C, i8* %"C'", i8* %A, i8* %"A'", i8* %B, i8* %"B'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.copy.garbage = alloca i8, align 1
; CHECK-NEXT:   %byref.copy.garbage3 = alloca i8, align 1
; CHECK-NEXT:   %byref.transa = alloca i8, align 1
; CHECK-NEXT:   %byref.transb = alloca i8, align 1
; CHECK-NEXT:   %byref.m = alloca i64, align 8
; CHECK-NEXT:   %byref.n = alloca i64, align 8
; CHECK-NEXT:   %byref.k = alloca i64, align 8
; CHECK-NEXT:   %byref.alpha = alloca double, align 8
; CHECK-NEXT:   %byref.lda = alloca i64, align 8
; CHECK-NEXT:   %byref.ldb = alloca i64, align 8
; CHECK-NEXT:   %byref.beta = alloca double, align 8
; CHECK-NEXT:   %byref.ldc = alloca i64, align 8
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8, align 1
; CHECK-NEXT:   %byref.transpose.transb = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.04 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.int.05 = alloca i64, align 8
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %alpha = alloca double, align 16
; CHECK-NEXT:   %alpha_p = bitcast double* %alpha to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %ldb = alloca i64, align 16
; CHECK-NEXT:   %ldb_p = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %beta = alloca double, align 16
; CHECK-NEXT:   %beta_p = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta, align 8
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %avld.transa = load i8, i8* %transa, align 1
; CHECK-NEXT:   %avld.transb = load i8, i8* %transb, align 1
; CHECK-NEXT:   %pcld.m = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %avld.m = load i64, i64* %pcld.m, align 4
; CHECK-NEXT:   %pcld.n = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %avld.n = load i64, i64* %pcld.n, align 4
; CHECK-NEXT:   %pcld.k = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %avld.k = load i64, i64* %pcld.k, align 4
; CHECK-NEXT:   %pcld.alpha = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %avld.alpha = load double, double* %pcld.alpha, align 8
; CHECK-NEXT:   %pcld.lda = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %avld.lda = load i64, i64* %pcld.lda, align 4
; CHECK-NEXT:   %pcld.ldb = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %avld.ldb = load i64, i64* %pcld.ldb, align 4
; CHECK-NEXT:   %pcld.beta = bitcast i8* %beta_p to double*
; CHECK-NEXT:   %avld.beta = load double, double* %pcld.beta, align 8
; CHECK-NEXT:   %pcld.ldc = bitcast i8* %ldc_p to i64*
; CHECK-NEXT:   %avld.ldc = load i64, i64* %pcld.ldc, align 4
; CHECK-NEXT:   %0 = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %1 = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %2 = load i64, i64* %0, align 4
; CHECK-NEXT:   %3 = load i64, i64* %1, align 4
; CHECK-NEXT:   %4 = mul i64 %2, %3
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %4, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %5 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage, align 1
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %m_p, i8* %k_p, i8* %A, i8* %lda_p, double* %5, i8* %m_p)
; CHECK-NEXT:   %6 = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %7 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %8 = load i64, i64* %6, align 4
; CHECK-NEXT:   %9 = load i64, i64* %7, align 4
; CHECK-NEXT:   %10 = mul i64 %8, %9
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %10, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %11 = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage3, align 1
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage3, i8* %k_p, i8* %n_p, i8* %B, i8* %ldb_p, double* %11, i8* %k_p)
; CHECK-NEXT:   %12 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } undef, i8 %avld.transa, 0
; CHECK-NEXT:   %13 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %12, i8 %avld.transb, 1
; CHECK-NEXT:   %14 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %13, i64 %avld.m, 2
; CHECK-NEXT:   %15 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %14, i64 %avld.n, 3
; CHECK-NEXT:   %16 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %15, i64 %avld.k, 4
; CHECK-NEXT:   %17 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %16, double %avld.alpha, 5
; CHECK-NEXT:   %18 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %17, i64 %avld.lda, 6
; CHECK-NEXT:   %19 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %18, i64 %avld.ldb, 7
; CHECK-NEXT:   %20 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %19, double %avld.beta, 8
; CHECK-NEXT:   %21 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %20, i64 %avld.ldc, 9
; CHECK-NEXT:   %22 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %21, double* %5, 10
; CHECK-NEXT:   %23 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %22, double* %11, 11
; CHECK-NEXT:   call void @dgemm_64_(i8* noundef nonnull %transa, i8* noundef nonnull %transb, i8* noundef nonnull %m_p, i8* noundef nonnull %n_p, i8* noundef nonnull %k_p, i8* noundef nonnull %alpha_p, i8* %A, i8* noundef nonnull %lda_p, i8* %B, i8* noundef nonnull %ldb_p, i8* noundef nonnull %beta_p, i8* %C, i8* noundef nonnull %ldc_p) #1
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %tape.ext.transa = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 0
; CHECK-NEXT:   store i8 %tape.ext.transa, i8* %byref.transa, align 1
; CHECK-NEXT:   %tape.ext.transb = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 1
; CHECK-NEXT:   store i8 %tape.ext.transb, i8* %byref.transb, align 1
; CHECK-NEXT:   %tape.ext.m = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 2
; CHECK-NEXT:   store i64 %tape.ext.m, i64* %byref.m, align 4
; CHECK-NEXT:   %cast.m = bitcast i64* %byref.m to i8*
; CHECK-NEXT:   %tape.ext.n = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 3
; CHECK-NEXT:   store i64 %tape.ext.n, i64* %byref.n, align 4
; CHECK-NEXT:   %cast.n = bitcast i64* %byref.n to i8*
; CHECK-NEXT:   %tape.ext.k = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 4
; CHECK-NEXT:   store i64 %tape.ext.k, i64* %byref.k, align 4
; CHECK-NEXT:   %cast.k = bitcast i64* %byref.k to i8*
; CHECK-NEXT:   %tape.ext.alpha = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 5
; CHECK-NEXT:   store double %tape.ext.alpha, double* %byref.alpha, align 8
; CHECK-NEXT:   %cast.alpha = bitcast double* %byref.alpha to i8*
; CHECK-NEXT:   %tape.ext.lda = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 6
; CHECK-NEXT:   store i64 %tape.ext.lda, i64* %byref.lda, align 4
; CHECK-NEXT:   %cast.lda = bitcast i64* %byref.lda to i8*
; CHECK-NEXT:   %tape.ext.ldb = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 7
; CHECK-NEXT:   store i64 %tape.ext.ldb, i64* %byref.ldb, align 4
; CHECK-NEXT:   %cast.ldb = bitcast i64* %byref.ldb to i8*
; CHECK-NEXT:   %tape.ext.beta = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 8
; CHECK-NEXT:   store double %tape.ext.beta, double* %byref.beta, align 8
; CHECK-NEXT:   %cast.beta = bitcast double* %byref.beta to i8*
; CHECK-NEXT:   %tape.ext.ldc = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 9
; CHECK-NEXT:   store i64 %tape.ext.ldc, i64* %byref.ldc, align 4
; CHECK-NEXT:   %cast.ldc = bitcast i64* %byref.ldc to i8*
; CHECK-NEXT:   %tape.ext.A = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 10
; CHECK-NEXT:   %24 = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   %tape.ext.B = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64, double*, double* } %23, 11
; CHECK-NEXT:   %25 = bitcast double* %tape.ext.B to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %byref.transa, align 1
; CHECK-NEXT:   %26 = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %27 = select i1 %26, i8 116, i8 0
; CHECK-NEXT:   %28 = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %29 = select i1 %28, i8 84, i8 %27
; CHECK-NEXT:   %30 = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %31 = select i1 %30, i8 110, i8 %29
; CHECK-NEXT:   %32 = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %33 = select i1 %32, i8 78, i8 %31
; CHECK-NEXT:   store i8 %33, i8* %byref.transpose.transa, align 1
; CHECK-NEXT:   %ld.transb = load i8, i8* %byref.transb, align 1
; CHECK-NEXT:   %34 = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %35 = select i1 %34, i8 116, i8 0
; CHECK-NEXT:   %36 = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %37 = select i1 %36, i8 84, i8 %35
; CHECK-NEXT:   %38 = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %39 = select i1 %38, i8 110, i8 %37
; CHECK-NEXT:   %40 = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %41 = select i1 %40, i8 78, i8 %39
; CHECK-NEXT:   store i8 %41, i8* %byref.transpose.transb, align 1
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transa, i8* %byref.transpose.transb, i8* %cast.m, i8* %cast.k, i8* %cast.n, i8* %cast.alpha, i8* %"C'", i8* %cast.ldc, i8* %25, i8* %cast.ldb, i8* %cast.beta, i8* %"A'", i8* %cast.lda)
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %byref.transb, i8* %cast.k, i8* %cast.n, i8* %cast.m, i8* %cast.alpha, i8* %24, i8* %cast.lda, i8* %"C'", i8* %cast.ldc, i8* %cast.beta, i8* %"B'", i8* %cast.ldb)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G, align 1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.04, align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.05, align 4
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %byref.constant.int.04, double* %byref.constant.fp.1.0, i8* %cast.beta, i8* %cast.m, i8* %cast.n, i8* %"C'", i8* %cast.ldc, i64* %byref.constant.int.05)
; CHECK-NEXT:   tail call void @free(i8* nonnull %24)
; CHECK-NEXT:   tail call void @free(i8* nonnull %25)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
