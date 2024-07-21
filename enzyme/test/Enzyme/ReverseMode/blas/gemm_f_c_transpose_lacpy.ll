;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1  -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i64, i64) 

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
  store i8 84, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1) 
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
; CHECK-NEXT:   %[[byrefgarbage:.+]] = alloca i8
; CHECK-NEXT:   %[[byrefgarbage2:.+]] = alloca i8
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[byref_fp_1_014:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %[[int04:.+]] = alloca i64
; CHECK-NEXT:   %[[byref_fp_1_018:.+]] = alloca double
; CHECK-NEXT:   %[[tmp:.+]] = alloca i64
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
; CHECK-NEXT:   store i8 84, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %loaded.trans = load i8, i8* %transa
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[i2:.+]] = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = select i1 %[[i2]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[i4:.+]] = select i1 %[[i2]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %[[i3]] to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %[[i4]] to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %[[i9:.+]] = mul i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[i9]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %[[byrefgarbage]]
; CHECK-NEXT:   call void @dlacpy_64_(i8* %[[byrefgarbage]], i8* %[[i3]], i8* %[[i4]], i8* %A, i8* %lda_p, double* %cache.A, i8* %[[i3]])
; CHECK-NEXT:   %loaded.trans1 = load i8, i8* %transb
; CHECK-DAG:   %[[i10:.+]] = icmp eq i8 %loaded.trans1, 78
; CHECK-DAG:   %[[i11:.+]] = icmp eq i8 %loaded.trans1, 110
; CHECK-NEXT:   %[[i12:.+]] = or i1 %[[i11]], %[[i10]]
; CHECK-NEXT:   %[[i13:.+]] = select i1 %[[i12]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[i14:.+]] = select i1 %[[i12]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[i15:.+]] = bitcast i8* %[[i13]] to i64*
; CHECK-NEXT:   %[[i17:.+]] = load i64, i64* %[[i15]]
; CHECK-NEXT:   %[[i16:.+]] = bitcast i8* %[[i14]] to i64*
; CHECK-NEXT:   %[[i18:.+]] = load i64, i64* %[[i16]]
; CHECK-NEXT:   %[[i19:.+]] = mul i64 %[[i17]], %[[i18]]
; CHECK-NEXT:   %[[mallocsize1:.+]] = mul nuw nsw i64 %[[i19]], 8
; CHECK-NEXT:   %[[malloccall2:.+]] = tail call noalias nonnull i8* @malloc(i64 %[[mallocsize1]])
; CHECK-NEXT:   %cache.B = bitcast i8* %[[malloccall2]] to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage4
; CHECK-NEXT:   call void @dlacpy_64_(i8* %[[byrefgarbage2]], i8* %[[i13]], i8* %[[i14]], i8* %B, i8* %ldb_p, double* %cache.B, i8* %[[i13]])
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %[[i24:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   %[[i25:.+]] = bitcast double* %cache.B to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:   %[[i34:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:   %[[i35:.+]] = select i1 %[[i34]], i8 116, i8 78
; CHECK-DAG:   %[[i36:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:   %[[i37:.+]] = select i1 %[[i36]], i8 84, i8 %[[i35]]
; CHECK-DAG:   %[[i38:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:   %[[i39:.+]] = select i1 %[[i38]], i8 110, i8 %[[i37]]
; CHECK-DAG:   %[[i40:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:   %[[i41:.+]] = select i1 %[[i40]], i8 78, i8 %[[i39]]
; CHECK-NEXT:   store i8 %[[i41]], i8* %byref.transpose.transb
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[a38:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[a39:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[a40:.+]] = or i1 %[[a39]], %[[a38]]
; CHECK-NEXT:   %[[a41:.+]] = select i1 %[[a40]], i8* %transa, i8* %transb
; CHECK-NEXT:   %[[a42:.+]] = select i1 %[[a40]], i8* %byref.transpose.transb, i8* %transa
; CHECK-NEXT:   %[[a43:.+]] = select i1 %[[a40]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[a44:.+]] = select i1 %[[a40]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %loaded.trans5 = load i8, i8* %transb
; CHECK-DAG:   %[[i40:.+]] = icmp eq i8 %loaded.trans5, 78
; CHECK-DAG:   %[[i41:.+]] = icmp eq i8 %loaded.trans5, 110
; CHECK-NEXT:   %[[i42:.+]] = or i1 %[[i41]], %[[i40]]
; CHECK-NEXT:   %[[i43:.+]] = select i1 %[[i42]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %loaded.trans6 = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[a49:.+]] = icmp eq i8 %loaded.trans6, 78
; CHECK-NEXT:   %[[a50:.+]] = icmp eq i8 %loaded.trans6, 110
; CHECK-NEXT:   %[[a51:.+]] = or i1 %[[a50]], %[[a49]]
; CHECK-NEXT:   %[[a52:.+]] = select i1 %[[a51]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %ld.row.trans7 = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[a53:.+]] = icmp eq i8 %ld.row.trans7, 110
; CHECK-NEXT:   %[[a54:.+]] = icmp eq i8 %ld.row.trans7, 78
; CHECK-NEXT:   %[[a55:.+]] = or i1 %[[a54]], %[[a53]]
; CHECK-NEXT:   %[[a56:.+]] = select i1 %[[a55]], i8* %"C'", i8* %[[i25]]
; CHECK-NEXT:   %[[a57:.+]] = select i1 %[[a55]], i8* %ldc_p, i8* %[[i43]]
; CHECK-NEXT:   %[[a58:.+]] = select i1 %[[a55]], i8* %[[i25]], i8* %"C'"
; CHECK-NEXT:   %[[a59:.+]] = select i1 %[[a55]], i8* %[[a52]], i8* %ldc_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[a41]], i8* %[[a42]], i8* %[[a43]], i8* %[[a44]], i8* %n_p, i8* %alpha_p, i8* %[[a56]], i8* %[[a57]], i8* %[[a58]], i8* %[[a59]], i8* %fpcast.constant.fp.1.0, i8* %"A'", i8* %lda_p, i64 1, i64 1)
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:   %[[i26:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:   %[[i27:.+]] = select i1 %[[i26]], i8 116, i8 78
; CHECK-DAG:   %[[i28:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:   %[[i29:.+]] = select i1 %[[i28]], i8 84, i8 %[[i27]]
; CHECK-DAG:   %[[i30:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:   %[[i31:.+]] = select i1 %[[i30]], i8 110, i8 %[[i29]]
; CHECK-DAG:   %[[i32:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:   %[[i33:.+]] = select i1 %[[i32]], i8 78, i8 %[[i31]]
; CHECK-NEXT:   store i8 %[[i33]], i8* %byref.transpose.transa
; CHECK-NEXT:   %[[ld_row_trans10:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[a60:.+]] = icmp eq i8 %[[ld_row_trans10]], 110
; CHECK-NEXT:   %[[a61:.+]] = icmp eq i8 %[[ld_row_trans10]], 78
; CHECK-NEXT:   %[[a62:.+]] = or i1 %[[a61]], %[[a60]]
; CHECK-NEXT:   %[[a63:.+]] = select i1 %[[a62]], i8* %byref.transpose.transa, i8* %transb
; CHECK-NEXT:   %[[a64:.+]] = select i1 %[[a62]], i8* %transb, i8* %transa
; CHECK-NEXT:   %[[a65:.+]] = select i1 %[[a62]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[a66:.+]] = select i1 %[[a62]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[cachedtrans2:.+]] = load i8, i8* %transa
; CHECK-DAG:   %[[i54:.+]] = icmp eq i8 %[[cachedtrans2]], 78
; CHECK-DAG:   %[[i55:.+]] = icmp eq i8 %[[cachedtrans2]], 110
; CHECK-NEXT:   %[[i56:.+]] = or i1 %[[i55]], %[[i54]]
; CHECK-NEXT:   %[[i57:.+]] = select i1 %[[i56]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans12:.+]] = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[a71:.+]] = icmp eq i8 %[[loaded_trans12]], 78
; CHECK-NEXT:   %[[a72:.+]] = icmp eq i8 %[[loaded_trans12]], 110
; CHECK-NEXT:   %[[a73:.+]] = or i1 %[[a72]], %[[a71]]
; CHECK-NEXT:   %[[a74:.+]] = select i1 %[[a73]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[ld_row_trans13:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[a75:.+]] = icmp eq i8 %[[ld_row_trans13]], 110
; CHECK-NEXT:   %[[a76:.+]] = icmp eq i8 %[[ld_row_trans13]], 78
; CHECK-NEXT:   %[[a77:.+]] = or i1 %[[a76]], %[[a75]]
; CHECK-NEXT:   %[[a78:.+]] = select i1 %[[a77]], i8* %[[i24]], i8* %"C'"
; CHECK-NEXT:   %[[a79:.+]] = select i1 %[[a77]], i8* %[[a74]], i8* %ldc_p
; CHECK-NEXT:   %[[a80:.+]] = select i1 %[[a77]], i8* %"C'", i8* %[[i24]]
; CHECK-NEXT:   %[[a81:.+]] = select i1 %[[a77]], i8* %ldc_p, i8* %[[i57]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_014]], align 8
; CHECK-NEXT:   %[[fpcast_1_015:.+]] = bitcast double* %[[byref_fp_1_014]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[a63]], i8* %[[a64]], i8* %[[a65]], i8* %[[a66]], i8* %m_p, i8* %alpha_p, i8* %[[a78]], i8* %[[a79]], i8* %[[a80]], i8* %[[a81]], i8* %[[fpcast_1_015]], i8* %"B'", i8* %ldb_p, i64 1, i64 1)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[int04]]
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %[[fp19:.+]] = bitcast double* %[[byref_fp_1_018]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[int04]], i8* %[[fp19]], i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   %[[free1:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[free1]])
; CHECK-NEXT:   %[[free2:.+]] = bitcast double* %cache.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[free2]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
