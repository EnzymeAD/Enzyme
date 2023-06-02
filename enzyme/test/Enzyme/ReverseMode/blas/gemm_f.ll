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
; CHECK-NEXT:   %avld.m = load i64, i64* %pcld.m
; CHECK-NEXT:   %pcld.n = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %avld.n = load i64, i64* %pcld.n
; CHECK-NEXT:   %pcld.k = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %avld.k = load i64, i64* %pcld.k
; CHECK-NEXT:   %pcld.alpha = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %avld.alpha = load double, double* %pcld.alpha
; CHECK-NEXT:   %pcld.lda = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %avld.lda = load i64, i64* %pcld.lda
; CHECK-NEXT:   %pcld.ldb = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %avld.ldb = load i64, i64* %pcld.ldb
; CHECK-NEXT:   %pcld.beta = bitcast i8* %beta_p to double*
; CHECK-NEXT:   %avld.beta = load double, double* %pcld.beta
; CHECK-NEXT:   %pcld.ldc = bitcast i8* %ldc_p to i64*
; CHECK-NEXT:   %avld.ldc = load i64, i64* %pcld.ldc
; CHECK-NEXT:   %0 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } undef, i8 %avld.transa, 0
; CHECK-NEXT:   %1 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %0, i8 %avld.transb, 1
; CHECK-NEXT:   %2 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %1, i64 %avld.m, 2
; CHECK-NEXT:   %3 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %2, i64 %avld.n, 3
; CHECK-NEXT:   %4 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %3, i64 %avld.k, 4
; CHECK-NEXT:   %5 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %4, double %avld.alpha, 5
; CHECK-NEXT:   %6 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %5, i64 %avld.lda, 6
; CHECK-NEXT:   %7 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %6, i64 %avld.ldb, 7
; CHECK-NEXT:   %8 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %7, double %avld.beta, 8
; CHECK-NEXT:   %9 = insertvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %8, i64 %avld.ldc, 9
; CHECK-NEXT:   call void @dgemm_64_(i8* noundef nonnull %transa, i8* noundef nonnull %transb, i8* noundef nonnull %m_p, i8* noundef nonnull %n_p, i8* noundef nonnull %k_p, i8* noundef nonnull %alpha_p, i8* %A, i8* noundef nonnull %lda_p, i8* %B, i8* noundef nonnull %ldb_p, i8* noundef nonnull %beta_p, i8* %C, i8* noundef nonnull %ldc_p) #3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %tape.ext.transa = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 0
; CHECK-NEXT:   store i8 %tape.ext.transa, i8* %byref.transa, align 1
; CHECK-NEXT:   %tape.ext.transb = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 1
; CHECK-NEXT:   store i8 %tape.ext.transb, i8* %byref.transb, align 1
; CHECK-NEXT:   %tape.ext.m = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 2
; CHECK-NEXT:   store i64 %tape.ext.m, i64* %byref.m, align 8
; CHECK-NEXT:   %cast.m = bitcast i64* %byref.m to i8*
; CHECK-NEXT:   %tape.ext.n = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 3
; CHECK-NEXT:   store i64 %tape.ext.n, i64* %byref.n, align 8
; CHECK-NEXT:   %cast.n = bitcast i64* %byref.n to i8*
; CHECK-NEXT:   %tape.ext.k = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 4
; CHECK-NEXT:   store i64 %tape.ext.k, i64* %byref.k, align 8
; CHECK-NEXT:   %cast.k = bitcast i64* %byref.k to i8*
; CHECK-NEXT:   %tape.ext.alpha = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 5
; CHECK-NEXT:   store double %tape.ext.alpha, double* %byref.alpha, align 8
; CHECK-NEXT:   %cast.alpha = bitcast double* %byref.alpha to i8*
; CHECK-NEXT:   %tape.ext.lda = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 6
; CHECK-NEXT:   store i64 %tape.ext.lda, i64* %byref.lda, align 8
; CHECK-NEXT:   %cast.lda = bitcast i64* %byref.lda to i8*
; CHECK-NEXT:   %tape.ext.ldb = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 7
; CHECK-NEXT:   store i64 %tape.ext.ldb, i64* %byref.ldb, align 8
; CHECK-NEXT:   %cast.ldb = bitcast i64* %byref.ldb to i8*
; CHECK-NEXT:   %tape.ext.beta = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 8
; CHECK-NEXT:   store double %tape.ext.beta, double* %byref.beta, align 8
; CHECK-NEXT:   %cast.beta = bitcast double* %byref.beta to i8*
; CHECK-NEXT:   %tape.ext.ldc = extractvalue { i8, i8, i64, i64, i64, double, i64, i64, double, i64 } %9, 9
; CHECK-NEXT:   store i64 %tape.ext.ldc, i64* %byref.ldc, align 8
; CHECK-NEXT:   %cast.ldc = bitcast i64* %byref.ldc to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %byref.transa, align 1
; CHECK-NEXT:   %10 = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %11 = select i1 %10, i8 116, i8 0
; CHECK-NEXT:   %12 = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %13 = select i1 %12, i8 84, i8 %11
; CHECK-NEXT:   %14 = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %15 = select i1 %14, i8 110, i8 %13
; CHECK-NEXT:   %16 = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %17 = select i1 %16, i8 78, i8 %15
; CHECK-NEXT:   store i8 %17, i8* %byref.transpose.transa, align 1
; CHECK-NEXT:   %ld.transb = load i8, i8* %byref.transb, align 1
; CHECK-NEXT:   %18 = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %19 = select i1 %18, i8 116, i8 0
; CHECK-NEXT:   %20 = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %21 = select i1 %20, i8 84, i8 %19
; CHECK-NEXT:   %22 = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %23 = select i1 %22, i8 110, i8 %21
; CHECK-NEXT:   %24 = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %25 = select i1 %24, i8 78, i8 %23
; CHECK-NEXT:   store i8 %25, i8* %byref.transpose.transb, align 1
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transa, i8* %byref.transpose.transb, i8* %cast.m, i8* %cast.k, i8* %cast.n, i8* %cast.alpha, i8* %"C'", i8* %cast.ldc, i8* %B, i8* %cast.ldb, i8* %cast.beta, i8* %"A'", i8* %cast.lda)
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %byref.transb, i8* %cast.k, i8* %cast.n, i8* %cast.m, i8* %cast.alpha, i8* %A, i8* %cast.lda, i8* %"C'", i8* %cast.ldc, i8* %cast.beta, i8* %"B'", i8* %cast.ldb)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

