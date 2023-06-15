;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-lapack-copy=1 -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1  -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly) 

define void @f(i8* noalias %C, i8* noalias %A, i8* noalias %B) {
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
  store i64 8, i64* %lda, align 16
  store i64 16, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 8, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p) 
  %ptr = bitcast i8* %B to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}
; A rule:                 /* A     */ (b<"gemm"> $layout, $transa, transpose<"transb">, $m, $k, $n, $alpha, adj<"C">, $ldc, $B, $ldb, $beta, adj<"A">, $lda),

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_const", i8* %B)
  ret void
}

; CHECK: define internal void @diffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.copy.garbage = alloca i8
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %[[byrefint03:.+]] = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %[[byrefint04:.+]] = alloca i64
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
; CHECK-NEXT:   store i64 8, i64* %lda, align 16
; CHECK-NEXT:   store i64 16, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta
; CHECK-NEXT:   store i64 8, i64* %ldc, align 16
; CHECK-NEXT:   %trans_check = load i8, i8* %transa, align 1 
; CHECK-NEXT:   %trans_check1 = load i8, i8* %transb, align 1 
; CHECK-NEXT:   %0 = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %1 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %2 = load i64, i64* %0
; CHECK-NEXT:   %3 = load i64, i64* %1
; CHECK-NEXT:   %4 = mul i64 %2, %3
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %4, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.B = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %k_p, i8* %n_p, i8* %B, i8* %ldb_p, double* %cache.B, i8* %k_p)
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %ptr = bitcast i8* %B to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %5 = bitcast double* %cache.B to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-NEXT:   %6 = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %7 = select i1 %6, i8 116, i8 0
; CHECK-NEXT:   %8 = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %9 = select i1 %8, i8 84, i8 %7
; CHECK-NEXT:   %10 = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %11 = select i1 %10, i8 110, i8 %9
; CHECK-NEXT:   %12 = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %13 = select i1 %12, i8 78, i8 %11
; CHECK-NEXT:   store i8 %13, i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-NEXT:   %14 = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %15 = select i1 %14, i8 116, i8 0
; CHECK-NEXT:   %16 = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %17 = select i1 %16, i8 84, i8 %15
; CHECK-NEXT:   %18 = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %19 = select i1 %18, i8 110, i8 %17
; CHECK-NEXT:   %20 = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %21 = select i1 %20, i8 78, i8 %19
; CHECK-NEXT:   store i8 %21, i8* %byref.transpose.transb
; CHECK-NEXT:   %get.cached.ld.trans = load i8, i8* %transb
; CHECK-NEXT:   %22 = icmp eq i8 %get.cached.ld.trans, 78
; CHECK-NEXT:   %23 = select i1 %22, i8* %k_p, i8* %n_p
; CHECK-NEXT:   %24 = icmp eq i8 %get.cached.ld.trans, 110
; CHECK-NEXT:   %25 = select i1 %24, i8* %k_p, i8* %23
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %5, i8* %25, i8* %beta_p, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[byrefint03]]
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   store i64 0, i64* %[[byrefint04]]
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byrefint03]], double* %byref.constant.fp.1.0, i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[byrefint04]])
; CHECK-NEXT:   %26 = bitcast double* %cache.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %26)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
