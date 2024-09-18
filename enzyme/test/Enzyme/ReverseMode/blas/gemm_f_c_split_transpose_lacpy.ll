;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i64, i64) 

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
  ret void
}

define void @g(i8* noalias %C, i8* noalias %A, i8* noalias %B) {
entry:
  call void @f(i8* %C, i8* %A, i8* %B)
  %ptr = bitcast i8* %A to double*
  store double 0.000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*)* @g, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %B, i8* %dB)
  ret void
}

; CHECK: define internal double* @augmented_f(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B, i8* %"B'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca double*
; CHECK-NEXT:   %byref.copy.garbage = alloca i8
; CHECK-NEXT:   %ldc = alloca i64, i64 1, align 16
; CHECK-NEXT:   %1 = bitcast i64* %ldc to i8*
; CHECK-NEXT:   %beta = alloca double, i64 1, align 16
; CHECK-NEXT:   %2 = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldb = alloca i64, i64 1, align 16
; CHECK-NEXT:   %3 = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %4 = bitcast i64* %lda to i8*
; CHECK-NEXT:   %alpha = alloca double, i64 1, align 16
; CHECK-NEXT:   %5 = bitcast double* %alpha to i8*
; CHECK-NEXT:   %k = alloca i64, i64 1, align 16
; CHECK-NEXT:   %6 = bitcast i64* %k to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %7 = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %8 = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %malloccall1 = alloca i8, i64 1, align 1
; CHECK-NEXT:   %9 = bitcast i8* %8 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %7 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %10 to i8*
; CHECK-NEXT:   %11 = bitcast i8* %6 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %k_p = bitcast i64* %11 to i8*
; CHECK-NEXT:   %12 = bitcast i8* %5 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %alpha_p = bitcast double* %12 to i8*
; CHECK-NEXT:   %13 = bitcast i8* %4 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %13 to i8*
; CHECK-NEXT:   %14 = bitcast i8* %3 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %ldb_p = bitcast i64* %14 to i8*
; CHECK-NEXT:   %15 = bitcast i8* %2 to double*, !enzyme_caststack !5
; CHECK-NEXT:   %beta_p = bitcast double* %15 to i8*
; CHECK-NEXT:   %16 = bitcast i8* %1 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %ldc_p = bitcast i64* %16 to i8*
; CHECK-NEXT:   store i8 84, i8* %malloccall, align 1
; CHECK-NEXT:   store i8 78, i8* %malloccall1, align 1
; CHECK-NEXT:   store i64 4, i64* %9, align 16
; CHECK-NEXT:   store i64 4, i64* %10, align 16
; CHECK-NEXT:   store i64 8, i64* %11, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %12, align 16
; CHECK-NEXT:   store i64 4, i64* %13, align 16
; CHECK-NEXT:   store i64 8, i64* %14, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %15
; CHECK-NEXT:   store i64 4, i64* %16, align 16
; CHECK-NEXT:   %loaded.trans = load i8, i8* %malloccall
; CHECK-DAG:   %[[r17:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[r18:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %19 = or i1 %[[r18]], %[[r17]]
; CHECK-NEXT:   %20 = select i1 %19, i8* %m_p, i8* %k_p
; CHECK-NEXT:   %21 = select i1 %19, i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[r22:.+]] = bitcast i8* %20 to i64*
; CHECK-NEXT:   %[[r24:.+]] = load i64, i64* %[[r22]]
; CHECK-NEXT:   %[[r23:.+]] = bitcast i8* %21 to i64*
; CHECK-NEXT:   %[[r25:.+]] = load i64, i64* %[[r23]]
; CHECK-NEXT:   %26 = mul i64 %[[r24]], %[[r25]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %26, 8
; CHECK-NEXT:   %malloccall10 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall10 to double*
; CHECK-NEXT:   store double* %cache.A, double** %0
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %20, i8* %21, i8* %A, i8* %lda_p, double* %cache.A, i8* %20)
; CHECK-NEXT:   call void @dgemm_64_(i8* %malloccall, i8* %malloccall1, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   %[[ret:.+]] = load double*, double** %0
; CHECK-NEXT:   ret double* %[[ret]]
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B, i8* %"B'", double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[byref_fp_1_0:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %[[byref_int_0:.+]] = alloca i64, align 8
; CHECK-NEXT:   %[[byref_fp_1_011:.+]] = alloca double, align 8
; CHECK-NEXT:   %[[tmp:.+]] = alloca i64
; CHECK-NEXT:   %ldc = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i1:.+]] = bitcast i64* %ldc to i8*
; CHECK-NEXT:   %beta = alloca double, i64 1, align 16
; CHECK-NEXT:   %[[i2:.+]] = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldb = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i3:.+]] = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i4:.+]] = bitcast i64* %lda to i8*
; CHECK-NEXT:   %alpha = alloca double, i64 1, align 16
; CHECK-NEXT:   %[[i5:.+]] = bitcast double* %alpha to i8*
; CHECK-NEXT:   %k = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i6:.+]] = bitcast i64* %k to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i7:.+]] = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %[[i8:.+]] = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %malloccall1 = alloca i8, i64 1, align 1
; CHECK-NEXT:   %[[i9:.+]] = bitcast i8* %[[i8]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %[[i9]] to i8*
; CHECK-NEXT:   %[[i10:.+]] = bitcast i8* %[[i7]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %[[i10]] to i8*
; CHECK-NEXT:   %[[i11:.+]] = bitcast i8* %[[i6]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %k_p = bitcast i64* %[[i11]] to i8*
; CHECK-NEXT:   %[[i12:.+]] = bitcast i8* %[[i5]] to double*, !enzyme_caststack !5
; CHECK-NEXT:   %alpha_p = bitcast double* %[[i12]] to i8*
; CHECK-NEXT:   %[[i13:.+]] = bitcast i8* %[[i4]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %[[i13]] to i8*
; CHECK-NEXT:   %[[i14:.+]] = bitcast i8* %[[i3]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %ldb_p = bitcast i64* %[[i14]] to i8*
; CHECK-NEXT:   %[[i15:.+]] = bitcast i8* %[[i2]] to double*, !enzyme_caststack !5
; CHECK-NEXT:   %beta_p = bitcast double* %[[i15]] to i8*
; CHECK-NEXT:   %[[i16:.+]] = bitcast i8* %[[i1]] to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %ldc_p = bitcast i64* %[[i16]] to i8*
; CHECK-NEXT:   store i8 84, i8* %malloccall, align 1
; CHECK-NEXT:   store i8 78, i8* %malloccall1, align 1
; CHECK-NEXT:   store i64 4, i64* %[[i9]], align 16
; CHECK-NEXT:   store i64 4, i64* %[[i10]], align 16
; CHECK-NEXT:   store i64 8, i64* %[[i11]], align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %[[i12]], align 16
; CHECK-NEXT:   store i64 4, i64* %[[i13]], align 16
; CHECK-NEXT:   store i64 8, i64* %[[i14]], align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %[[i15]]
; CHECK-NEXT:   store i64 4, i64* %[[i16]], align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %[[r17:.+]] = bitcast double* %0 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %ld.transb = load i8, i8* %malloccall1
; CHECK-DAG:   %[[r26:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:   %[[r27:.+]] = select i1 %[[r26]], i8 116, i8 78
; CHECK-DAG:   %[[r28:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:   %[[r29:.+]] = select i1 %[[r28]], i8 84, i8 %[[r27]]
; CHECK-DAG:   %[[r30:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:   %[[r31:.+]] = select i1 %[[r30]], i8 110, i8 %[[r29]]
; CHECK-DAG:   %[[r32:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:   %[[r33:.+]] = select i1 %[[r32]], i8 78, i8 %[[r31]]
; CHECK-NEXT:   store i8 %[[r33]], i8* %byref.transpose.transb
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %malloccall, align 1
; CHECK-NEXT:   %[[r34:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[r35:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r36:.+]] = or i1 %[[r35]], %[[r34]]
; CHECK-NEXT:   %[[r37:.+]] = select i1 %[[r36]], i8* %malloccall, i8* %malloccall1
; CHECK-NEXT:   %[[r38:.+]] = select i1 %[[r36]], i8* %byref.transpose.transb, i8* %malloccall
; CHECK-NEXT:   %[[r39:.+]] = select i1 %[[r36]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[r40:.+]] = select i1 %[[r36]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %ld.row.trans1 = load i8, i8* %malloccall, align 1
; CHECK-NEXT:   %[[r41:.+]] = icmp eq i8 %ld.row.trans1, 110
; CHECK-NEXT:   %[[r42:.+]] = icmp eq i8 %ld.row.trans1, 78
; CHECK-NEXT:   %[[r43:.+]] = or i1 %[[r42]], %[[r41]]
; CHECK-NEXT:   %[[r44:.+]] = select i1 %[[r43]], i8* %"C'", i8* %B
; CHECK-NEXT:   %[[r45:.+]] = select i1 %[[r43]], i8* %ldc_p, i8* %ldb_p
; CHECK-NEXT:   %[[r46:.+]] = select i1 %[[r43]], i8* %B, i8* %"C'"
; CHECK-NEXT:   %[[r47:.+]] = select i1 %[[r43]], i8* %ldb_p, i8* %ldc_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r37]], i8* %[[r38]], i8* %[[r39]], i8* %[[r40]], i8* %n_p, i8* %alpha_p, i8* %[[r44]], i8* %[[r45]], i8* %[[r46]], i8* %[[r47]], i8* %fpcast.constant.fp.1.0, i8* %"A'", i8* %lda_p, i64 1, i64 1)
; CHECK-NEXT:   %ld.transa = load i8, i8* %malloccall
; CHECK-DAG:   %[[r18:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:   %[[r19:.+]] = select i1 %[[r18]], i8 116, i8 78
; CHECK-DAG:   %[[r20:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:   %[[r21:.+]] = select i1 %[[r20]], i8 84, i8 %[[r19]]
; CHECK-DAG:   %[[r22:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:   %[[r23:.+]] = select i1 %[[r22]], i8 110, i8 %[[r21]]
; CHECK-DAG:   %[[r24:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:   %[[r25:.+]] = select i1 %[[r24]], i8 78, i8 %[[r23]]
; CHECK-NEXT:   store i8 %[[r25]], i8* %byref.transpose.transa
; CHECK-NEXT:   %[[ld_row_trans4:.+]] = load i8, i8* %malloccall1, align 1
; CHECK-NEXT:   %[[r48:.+]] = icmp eq i8 %[[ld_row_trans4]], 110
; CHECK-NEXT:   %[[r49:.+]] = icmp eq i8 %[[ld_row_trans4]], 78
; CHECK-NEXT:   %[[r50:.+]] = or i1 %[[r49]], %[[r48]]
; CHECK-NEXT:   %[[r51:.+]] = select i1 %[[r50]], i8* %byref.transpose.transa, i8* %malloccall1
; CHECK-NEXT:   %[[r52:.+]] = select i1 %[[r50]], i8* %malloccall1, i8* %malloccall
; CHECK-NEXT:   %[[r53:.+]] = select i1 %[[r50]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[r54:.+]] = select i1 %[[r50]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %loaded.trans = load i8, i8* %malloccall, align 1
; CHECK-NEXT:   %[[r55:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-NEXT:   %[[r56:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[r57:.+]] = or i1 %[[r56]], %[[r55]]
; CHECK-NEXT:   %[[r58:.+]] = select i1 %[[r57]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans5:.+]] = load i8, i8* %malloccall, align 1
; CHECK-NEXT:   %[[r59:.+]] = icmp eq i8 %[[loaded_trans5]], 78
; CHECK-NEXT:   %[[r60:.+]] = icmp eq i8 %[[loaded_trans5]], 110
; CHECK-NEXT:   %[[r61:.+]] = or i1 %[[r60]], %[[r59]]
; CHECK-NEXT:   %[[r62:.+]] = select i1 %[[r61]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[ld_row_trans6:.+]] = load i8, i8* %malloccall1, align 1
; CHECK-NEXT:   %[[r63:.+]] = icmp eq i8 %[[ld_row_trans6]], 110
; CHECK-NEXT:   %[[r64:.+]] = icmp eq i8 %[[ld_row_trans6]], 78
; CHECK-NEXT:   %[[r65:.+]] = or i1 %[[r64]], %[[r63]]
; CHECK-NEXT:   %[[r66:.+]] = select i1 %[[r65]], i8* %[[r17]], i8* %"C'"
; CHECK-NEXT:   %[[r67:.+]] = select i1 %[[r65]], i8* %[[r62]], i8* %ldc_p
; CHECK-NEXT:   %[[r68:.+]] = select i1 %[[r65]], i8* %"C'", i8* %[[r17]]
; CHECK-NEXT:   %[[r69:.+]] = select i1 %[[r65]], i8* %ldc_p, i8* %[[r58]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_0]], align 8
; CHECK-NEXT:   %[[fpcast_1_08:.+]] = bitcast double* %[[byref_fp_1_0]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r51]], i8* %[[r52]], i8* %[[r53]], i8* %[[r54]], i8* %m_p, i8* %alpha_p, i8* %[[r66]], i8* %[[r67]], i8* %[[r68]], i8* %[[r69]], i8* %[[fpcast_1_08]], i8* %"B'", i8* %ldb_p, i64 1, i64 1)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G, align 1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   store i64 0, i64* %[[byref_int_0]], align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_011]], align 8
; CHECK-NEXT:   %[[fpcast_1_012:.+]] = bitcast double* %[[byref_fp_1_011]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byref_int_0]], i8* %[[fpcast_1_012]], i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   %[[r70:.+]] = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r70]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
