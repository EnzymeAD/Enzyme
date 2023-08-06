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
; CHECK-NEXT:   %byref.int.one = alloca i64
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
; CHECK-NEXT:   %loaded.trans = load i8, i8* %transb
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %2 = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %3 = select i1 %2, i8* %k_p, i8* %n_p
; CHECK-NEXT:   %4 = select i1 %2, i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %3 to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %4 to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %9 = mul i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %9, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.B = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %3, i8* %4, i8* %B, i8* %ldb_p, double* %cache.B, i8* %3)
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %ptr = bitcast i8* %B to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %10 = bitcast double* %cache.B to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 0
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-DAG:    store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[r8:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[r9:.+]] = select i1 %[[r8]], i8 116, i8 0
; CHECK-DAG:    %[[r10:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[r11:.+]] = select i1 %[[r10]], i8 84, i8 %[[r9]]
; CHECK-DAG:    %[[r12:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[r13:.+]] = select i1 %[[r12]], i8 110, i8 %[[r11]]
; CHECK-DAG:    %[[r14:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[r15:.+]] = select i1 %[[r14]], i8 78, i8 %[[r13]]
; CHECK-DAG:    store i8 %[[r15]], i8* %byref.transpose.transb
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   %loaded.trans1 = load i8, i8* %transb
; CHECK-DAG:   %[[r16:.+]] = icmp eq i8 %loaded.trans1, 78
; CHECK-DAG:   %[[r17:.+]] = icmp eq i8 %loaded.trans1, 110
; CHECK-NEXT:   %[[r18:.+]] = or i1 %[[r17]], %[[r16]]
; CHECK-NEXT:   %[[r19:.+]] = select i1 %[[r18]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %10, i8* %[[r19]], i8* %beta_p, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   %intcast.constant.int.0 = bitcast i64* %byref.constant.int.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.02
; CHECK-NEXT:   %intcast.constant.int.03 = bitcast i64* %byref.constant.int.02 to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.04
; CHECK-NEXT:   %intcast.constant.int.05 = bitcast i64* %byref.constant.int.04 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i8* %intcast.constant.int.0, i8* %intcast.constant.int.03, i8* %fpcast.constant.fp.1.0, i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i8* %intcast.constant.int.05)
; CHECK-NEXT:   %[[ret:.+]] = bitcast double* %cache.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[ret]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
