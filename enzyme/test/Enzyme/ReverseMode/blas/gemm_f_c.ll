;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

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
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[byref_fp_1_013:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %[[byrefconstantint4:.+]] = alloca i64, align 8
; CHECK-NEXT:   %[[byref_fp_1_017:.+]] = alloca double, align 8
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
; CHECK-NEXT:   %[[i10:.+]] = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %[[i11:.+]] = load i64, i64* %[[i10]]
; CHECK-NEXT:   %[[i12:.+]] = bitcast i8* %A to double*
; CHECK:   %mul.i = add nuw nsw i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %[[i13:.+]] = icmp eq i64 %mul.i, 0
; CHECK-NEXT:   br i1 %[[i13]], label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %init.end.i, %entry
; CHECK-NEXT:   %j.i = phi i64 [ 0, %entry ], [ %j.next.i, %init.end.i ]
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %i.i = phi i64 [ 0, %init.idx.i ], [ %i.next.i, %for.body.i ]
; CHECK-NEXT:   %[[i14:.+]] = mul nuw nsw i64 %j.i, %[[i7]]
; CHECK-NEXT:   %[[i15:.+]] = add nuw nsw i64 %i.i, %[[i14]]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %cache.A, i64 %[[i15]]
; CHECK-NEXT:   %[[i16:.+]] = mul nuw nsw i64 %j.i, %[[i11]]
; CHECK-NEXT:   %[[i17:.+]] = add nuw nsw i64 %i.i, %[[i16]]
; CHECK-NEXT:   %dst.i1.i = getelementptr inbounds double, double* %[[i12]], i64 %[[i17]]
; CHECK-NEXT:   %src.i.l.i = load double, double* %dst.i1.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %i.next.i = add nuw nsw i64 %i.i, 1
; CHECK-NEXT:   %[[i18:.+]] = icmp eq i64 %i.next.i, %[[i7]]
; CHECK-NEXT:   br i1 %[[i18]], label %init.end.i, label %for.body.i

; CHECK: init.end.i:                                       ; preds = %for.body.i
; CHECK-NEXT:   %j.next.i = add nuw nsw i64 %j.i, 1
; CHECK-NEXT:   %[[i19:.+]] = icmp eq i64 %j.next.i, %[[i8]]
; CHECK-NEXT:   br i1 %[[i19]], label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: __enzyme_memcpy_double_mat_64.exit:               ; preds = %entry, %init.end.i
; CHECK-NEXT:   %loaded.trans1 = load i8, i8* %transb
; CHECK-DAG:   %[[i20:.+]] = icmp eq i8 %loaded.trans1, 78
; CHECK-DAG:   %[[i21:.+]] = icmp eq i8 %loaded.trans1, 110
; CHECK-NEXT:   %[[i22:.+]] = or i1 %[[i21]], %[[i20]]
; CHECK-NEXT:   %[[i23:.+]] = select i1 %[[i22]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[i24:.+]] = select i1 %[[i22]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[i25:.+]] = bitcast i8* %[[i23]] to i64*
; CHECK-NEXT:   %[[i27:.+]] = load i64, i64* %[[i25]]
; CHECK-NEXT:   %[[i26:.+]] = bitcast i8* %[[i24]] to i64*
; CHECK-NEXT:   %[[i28:.+]] = load i64, i64* %[[i26]]
; CHECK-NEXT:   %[[i29:.+]] = mul i64 %[[i27]], %[[i28]]
; CHECK-NEXT:   %mallocsize2 = mul nuw nsw i64 %[[i29]], 8
; CHECK-NEXT:   %malloccall3 = tail call noalias nonnull i8* @malloc(i64 %mallocsize2)
; CHECK-NEXT:   %cache.B = bitcast i8* %malloccall3 to double*
; CHECK-NEXT:   %[[i30:.+]] = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %[[i31:.+]] = load i64, i64* %[[i30]]
; CHECK-NEXT:   %[[i32:.+]] = bitcast i8* %B to double*
; CHECK:   %[[mul_i8:.+]] = add nuw nsw i64 %[[i27]], %[[i28]]
; CHECK-NEXT:   %[[i33:.+]] = icmp eq i64 %[[mul_i8]], 0
; CHECK-NEXT:   br i1 %[[i33]], label %[[enzyme_memcpy_double_mat_64_exit21:.+]], label %[[init_idx:.+]]

; CHECK: [[init_idx]]:                                      ; preds = %[[init_end_i18:.+]], %__enzyme_memcpy_double_mat_64.exit
; CHECK-NEXT:   %[[j_i9:.+]] = phi i64 [ 0, %__enzyme_memcpy_double_mat_64.exit ], [ %[[j_next_i17:.+]], %[[init_end_i18]] ]
; CHECK-NEXT:   br label %[[for_body_i16:.+]]

; CHECK: [[for_body_i16]]:                                     ; preds = %[[for_body_i16]], %[[init_idx]]
; CHECK-NEXT:   %[[i_i11:.+]] = phi i64 [ 0, %[[init_idx]] ], [ %[[i_next_i15:.+]], %[[for_body_i16]] ]
; CHECK-NEXT:   %[[i34:.+]] = mul nuw nsw i64 %[[j_i9]], %[[i27]]
; CHECK-NEXT:   %[[i35:.+]] = add nuw nsw i64 %[[i_i11]], %[[i34]]
; CHECK-NEXT:   %[[dst_i_i12:.+]] = getelementptr inbounds double, double* %cache.B, i64 %[[i35]]
; CHECK-NEXT:   %[[i36:.+]] = mul nuw nsw i64 %[[j_i9]], %[[i31]]
; CHECK-NEXT:   %[[i37:.+]] = add nuw nsw i64 %[[i_i11]], %[[i36]]
; CHECK-NEXT:   %[[dst_i1_i13:.+]] = getelementptr inbounds double, double* %[[i32]], i64 %[[i37]]
; CHECK-NEXT:   %[[src_i_l_i14:.+]] = load double, double* %[[dst_i1_i13]]
; CHECK-NEXT:   store double %[[src_i_l_i14]], double* %[[dst_i_i12]]
; CHECK-NEXT:   %[[i_next_i15]] = add nuw nsw i64 %[[i_i11]], 1
; CHECK-NEXT:   %[[i38:.+]] = icmp eq i64 %[[i_next_i15]], %[[i27]]
; CHECK-NEXT:   br i1 %[[i38]], label %[[init_end_i18]], label %[[for_body_i16]]

; CHECK: [[init_end_i18]]:                                     
; CHECK-NEXT:   %[[j_next_i17]] = add nuw nsw i64 %[[j_i9]], 1
; CHECK-NEXT:   %[[i39:.+]] = icmp eq i64 %[[j_next_i17]], %[[i28]]
; CHECK-NEXT:   br i1 %[[i39]], label %[[enzyme_memcpy_double_mat_64_exit21]], label %[[init_idx]]

; CHECK: [[enzyme_memcpy_double_mat_64_exit21]]:             ; preds = %__enzyme_memcpy_double_mat_64.exit, %[[init_end_i18]]
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %[[enzyme_memcpy_double_mat_64_exit21]]
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %[[i42:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   %[[i43:.+]] = bitcast double* %cache.B to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[r8:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[r9:.+]] = select i1 %[[r8]], i8 116, i8 78
; CHECK-DAG:    %[[r10:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[r11:.+]] = select i1 %[[r10]], i8 84, i8 %[[r9]]
; CHECK-DAG:    %[[r12:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[r13:.+]] = select i1 %[[r12]], i8 110, i8 %[[r11]]
; CHECK-DAG:    %[[r14:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[r15:.+]] = select i1 %[[r14]], i8 78, i8 %[[r13]]
; CHECK-NEXT:   store i8 %[[r15]], i8* %byref.transpose.transb
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r58:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[r59:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r60:.+]] = or i1 %[[r59]], %[[r58]]
; CHECK-NEXT:   %[[r61:.+]] = select i1 %[[r60]], i8* %transa, i8* %transb
; CHECK-NEXT:   %[[r62:.+]] = select i1 %[[r60]], i8* %byref.transpose.transb, i8* %transa
; CHECK-NEXT:   %[[r63:.+]] = select i1 %[[r60]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[r64:.+]] = select i1 %[[r60]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %loaded.trans4 = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r65:.+]] = icmp eq i8 %loaded.trans4, 78
; CHECK-NEXT:   %[[r66:.+]] = icmp eq i8 %loaded.trans4, 110
; CHECK-NEXT:   %[[r67:.+]] = or i1 %[[r66]], %[[r65]]
; CHECK-NEXT:   %[[r68:.+]] = select i1 %[[r67]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %loaded.trans5 = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r69:.+]] = icmp eq i8 %loaded.trans5, 78
; CHECK-NEXT:   %[[r70:.+]] = icmp eq i8 %loaded.trans5, 110
; CHECK-NEXT:   %[[r71:.+]] = or i1 %[[r70]], %[[r69]]
; CHECK-NEXT:   %[[r72:.+]] = select i1 %[[r71]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %ld.row.trans6 = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r73:.+]] = icmp eq i8 %ld.row.trans6, 110
; CHECK-NEXT:   %[[r74:.+]] = icmp eq i8 %ld.row.trans6, 78
; CHECK-NEXT:   %[[r75:.+]] = or i1 %[[r74]], %[[r73]]
; CHECK-NEXT:   %[[r76:.+]] = select i1 %[[r75]], i8* %"C'", i8* %[[i43]]
; CHECK-NEXT:   %[[r77:.+]] = select i1 %[[r75]], i8* %ldc_p, i8* %[[r68]]
; CHECK-NEXT:   %[[r78:.+]] = select i1 %[[r75]], i8* %[[i43]], i8* %"C'"
; CHECK-NEXT:   %[[r79:.+]] = select i1 %[[r75]], i8* %[[r72]], i8* %ldc_p
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r61]], i8* %[[r62]], i8* %[[r63]], i8* %[[r64]], i8* %n_p, i8* %alpha_p, i8* %[[r76]], i8* %[[r77]], i8* %[[r78]], i8* %[[r79]], i8* %fpcast.constant.fp.1.0, i8* %"A'", i8* %lda_p, i64 1, i64 1)
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 78
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   %[[ld_row_trans9:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r80:.+]] = icmp eq i8 %[[ld_row_trans9]], 110
; CHECK-NEXT:   %[[r81:.+]] = icmp eq i8 %[[ld_row_trans9]], 78
; CHECK-NEXT:   %[[r82:.+]] = or i1 %[[r81]], %[[r80]]
; CHECK-NEXT:   %[[r83:.+]] = select i1 %[[r82]], i8* %byref.transpose.transa, i8* %transb
; CHECK-NEXT:   %[[r84:.+]] = select i1 %[[r82]], i8* %transb, i8* %transa
; CHECK-NEXT:   %[[r85:.+]] = select i1 %[[r82]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[r86:.+]] = select i1 %[[r82]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans10:.+]] = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r87:.+]] = icmp eq i8 %[[loaded_trans10]], 78
; CHECK-NEXT:   %[[r88:.+]] = icmp eq i8 %[[loaded_trans10]], 110
; CHECK-NEXT:   %[[r89:.+]] = or i1 %[[r88]], %[[r87]]
; CHECK-NEXT:   %[[r90:.+]] = select i1 %[[r89]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans11:.+]] = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r91:.+]] = icmp eq i8 %[[loaded_trans11]], 78
; CHECK-NEXT:   %[[r92:.+]] = icmp eq i8 %[[loaded_trans11]], 110
; CHECK-NEXT:   %[[r93:.+]] = or i1 %[[r92]], %[[r91]]
; CHECK-NEXT:   %[[r94:.+]] = select i1 %[[r93]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[ld_row_trans12:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r95:.+]] = icmp eq i8 %[[ld_row_trans12]], 110
; CHECK-NEXT:   %[[r96:.+]] = icmp eq i8 %[[ld_row_trans12]], 78
; CHECK-NEXT:   %[[r97:.+]] = or i1 %[[r96]], %[[r95]]
; CHECK-NEXT:   %[[r98:.+]] = select i1 %[[r97]], i8* %[[i42]], i8* %"C'"
; CHECK-NEXT:   %[[r99:.+]] = select i1 %[[r97]], i8* %[[r94]], i8* %ldc_p
; CHECK-NEXT:   %[[r100:.+]] = select i1 %[[r97]], i8* %"C'", i8* %[[i42]]
; CHECK-NEXT:   %[[r101:.+]] = select i1 %[[r97]], i8* %ldc_p, i8* %[[r90]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_013]], align 8
; CHECK-NEXT:   %[[fpcast_1_014:.+]] = bitcast double* %[[byref_fp_1_013]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r83]], i8* %[[r84]], i8* %[[r85]], i8* %[[r86]], i8* %m_p, i8* %alpha_p, i8* %[[r98]], i8* %[[r99]], i8* %[[r100]], i8* %[[r101]], i8* %[[fpcast_1_014]], i8* %"B'", i8* %ldb_p, i64 1, i64 1)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[byrefconstantint4]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_017]]
; CHECK-NEXT:   %[[fpcast_1_018:.+]] = bitcast double* %[[byref_fp_1_017]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byrefconstantint4]], i8* %[[fpcast_1_018]], i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   %[[ret1:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[ret1]])
; CHECK-NEXT:   %[[ret2:.+]] = bitcast double* %cache.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[ret2]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
