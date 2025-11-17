;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S -enzyme-detect-readthrow=0 | FileCheck %s

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
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
  store i64 0, i64* %m, align 16
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
; CHECK-NEXT:   %byref.m = alloca i64
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[byref_fp_1:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %[[byref_int_0:.+]] = alloca i64, align 8
; CHECK-NEXT:   %[[byref_fp_1_0:.+]] = alloca double, align 8
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
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %pcld.m = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %avld.m = load i64, i64* %pcld.m
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   store i64 0, i64* %m
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 %avld.m, i64* %byref.m
; CHECK-NEXT:   %cast.m = bitcast i64* %byref.m to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[i18:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[i19:.+]] = select i1 %[[i18:.+]], i8 116, i8 78
; CHECK-DAG:    %[[i20:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[i21:.+]] = select i1 %[[i20:.+]], i8 84, i8 %[[i19]]
; CHECK-DAG:    %[[i22:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[i23:.+]] = select i1 %[[i22:.+]], i8 110, i8 %[[i21]]
; CHECK-DAG:    %[[i24:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[i25:.+]] = select i1 %[[i24:.+]], i8 78, i8 %[[i23]]
; CHECK-DAG:    store i8 %[[i25]], i8* %byref.transpose.transb
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r16:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[r17:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r18:.+]] = or i1 %[[r17]], %[[r16]]
; CHECK-NEXT:   %[[r19:.+]] = select i1 %[[r18]], i8* %transa, i8* %transb
; CHECK-NEXT:   %[[r20:.+]] = select i1 %[[r18]], i8* %byref.transpose.transb, i8* %transa
; CHECK-NEXT:   %[[r21:.+]] = select i1 %[[r18]], i8* %cast.m, i8* %k_p
; CHECK-NEXT:   %[[r22:.+]] = select i1 %[[r18]], i8* %k_p, i8* %cast.m
; CHECK-NEXT:   %ld.row.trans1 = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r23:.+]] = icmp eq i8 %ld.row.trans1, 110
; CHECK-NEXT:   %[[r24:.+]] = icmp eq i8 %ld.row.trans1, 78
; CHECK-NEXT:   %[[r25:.+]] = or i1 %[[r24]], %[[r23]]
; CHECK-NEXT:   %[[r26:.+]] = select i1 %[[r25:.+]], i8* %"C'", i8* %B
; CHECK-NEXT:   %[[r27:.+]] = select i1 %[[r25:.+]], i8* %ldc_p, i8* %ldb_p
; CHECK-NEXT:   %[[r28:.+]] = select i1 %[[r25:.+]], i8* %B, i8* %"C'"
; CHECK-NEXT:   %[[r29:.+]] = select i1 %[[r25:.+]], i8* %ldb_p, i8* %ldc_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r19]], i8* %[[r20]], i8* %[[r21]], i8* %[[r22]], i8* %n_p, i8* %alpha_p, i8* %[[r26]], i8* %[[r27]], i8* %[[r28]], i8* %[[r29]], i8* %fpcast.constant.fp.1.0, i8* %"A'", i8* %lda_p, i64 1, i64 1)
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[i10:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[i11:.+]] = select i1 %[[i10]], i8 116, i8 78
; CHECK-DAG:    %[[i12:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[i13:.+]] = select i1 %[[i12]], i8 84, i8 %[[i11]]
; CHECK-DAG:    %[[i14:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[i15:.+]] = select i1 %[[i14]], i8 110, i8 %[[i13]]
; CHECK-DAG:    %[[i16:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[i17:.+]] = select i1 %[[i16]], i8 78, i8 %[[i15]]
; CHECK-DAG:    store i8 %[[i17]], i8* %byref.transpose.transa
; CHECK-NEXT:   %[[ld_row_trans2:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r30:.+]] = icmp eq i8 %[[ld_row_trans2]], 110
; CHECK-NEXT:   %[[r31:.+]] = icmp eq i8 %[[ld_row_trans2]], 78
; CHECK-NEXT:   %[[r32:.+]] = or i1 %[[r31]], %[[r30]]
; CHECK-NEXT:   %[[r33:.+]] = select i1 %[[r32]], i8* %byref.transpose.transa, i8* %transb
; CHECK-NEXT:   %[[r34:.+]] = select i1 %[[r32]], i8* %transb, i8* %transa
; CHECK-NEXT:   %[[r35:.+]] = select i1 %[[r32]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[r36:.+]] = select i1 %[[r32]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[ld_row_trans3:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r37:.+]] = icmp eq i8 %[[ld_row_trans3]], 110
; CHECK-NEXT:   %[[r38:.+]] = icmp eq i8 %[[ld_row_trans3]], 78
; CHECK-NEXT:   %[[r39:.+]] = or i1 %[[r38]], %[[r37]]
; CHECK-NEXT:   %[[r40:.+]] = select i1 %[[r39]], i8* %A, i8* %"C'"
; CHECK-NEXT:   %[[r41:.+]] = select i1 %[[r39]], i8* %lda_p, i8* %ldc_p
; CHECK-NEXT:   %[[r42:.+]] = select i1 %[[r39]], i8* %"C'", i8* %A
; CHECK-NEXT:   %[[r43:.+]] = select i1 %[[r39]], i8* %ldc_p, i8* %lda_p
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1]], align 8
; CHECK-NEXT:   %[[fpcast_1_0:.+]] = bitcast double* %[[byref_fp_1]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r33]], i8* %[[r34]], i8* %[[r35]], i8* %[[r36]], i8* %cast.m, i8* %alpha_p, i8* %[[r40]], i8* %[[r41]], i8* %[[r42]], i8* %[[r43]], i8* %[[fpcast_1_0]], i8* %"B'", i8* %ldb_p, i64 1, i64 1)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G, align 1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   store i64 0, i64* %[[byref_int_0]], align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_0]], align 8
; CHECK-NEXT:   %[[fpcast_1:.+]] = bitcast double* %[[byref_fp_1_0]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byref_int_0]], i8* %[[fpcast_1]], i8* %beta_p, i8* %cast.m, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

