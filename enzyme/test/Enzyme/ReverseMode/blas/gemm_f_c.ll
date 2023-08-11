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
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p) 
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
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %[[byrefconstantint4:.+]] = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %[[byrefconstantint5:.+]] = alloca i64
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
; CHECK-NEXT:   %2 = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %3 = select i1 %2, i8* %m_p, i8* %k_p
; CHECK-NEXT:   %4 = select i1 %2, i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %3 to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %4 to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %9 = mul i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %9, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %10 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %11 = load i64, i64* %10
; CHECK-NEXT:   %12 = bitcast i8* %A to double*
; CHECK:   %mul.i = add nuw nsw i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %13 = icmp eq i64 %mul.i, 0
; CHECK-NEXT:   br i1 %13, label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %init.end.i, %entry
; CHECK-NEXT:   %j.i = phi i64 [ 0, %entry ], [ %j.next.i, %init.end.i ]
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %i.i = phi i64 [ 0, %init.idx.i ], [ %i.next.i, %for.body.i ]
; CHECK-NEXT:   %14 = mul nuw nsw i64 %j.i, %[[i7]]
; CHECK-NEXT:   %15 = add nuw nsw i64 %i.i, %14
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %cache.A, i64 %15
; CHECK-NEXT:   %16 = mul nuw nsw i64 %j.i, %11
; CHECK-NEXT:   %17 = add nuw nsw i64 %i.i, %16
; CHECK-NEXT:   %dst.i1.i = getelementptr inbounds double, double* %12, i64 %17
; CHECK-NEXT:   %src.i.l.i = load double, double* %dst.i1.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %i.next.i = add nuw nsw i64 %i.i, 1
; CHECK-NEXT:   %18 = icmp eq i64 %i.next.i, %[[i7]]
; CHECK-NEXT:   br i1 %18, label %init.end.i, label %for.body.i

; CHECK: init.end.i:                                       ; preds = %for.body.i
; CHECK-NEXT:   %j.next.i = add nuw nsw i64 %j.i, 1
; CHECK-NEXT:   %19 = icmp eq i64 %j.next.i, %[[i8]]
; CHECK-NEXT:   br i1 %19, label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: __enzyme_memcpy_double_mat_64.exit:               ; preds = %entry, %init.end.i
; CHECK-NEXT:   %loaded.trans1 = load i8, i8* %transb
; CHECK-DAG:   %[[i20:.+]] = icmp eq i8 %loaded.trans1, 78
; CHECK-DAG:   %[[i21:.+]] = icmp eq i8 %loaded.trans1, 110
; CHECK-NEXT:   %22 = or i1 %[[i21]], %[[i20]]
; CHECK-NEXT:   %23 = select i1 %22, i8* %k_p, i8* %n_p
; CHECK-NEXT:   %24 = select i1 %22, i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[i25:.+]] = bitcast i8* %23 to i64*
; CHECK-NEXT:   %[[i27:.+]] = load i64, i64* %[[i25]]
; CHECK-NEXT:   %[[i26:.+]] = bitcast i8* %24 to i64*
; CHECK-NEXT:   %[[i28:.+]] = load i64, i64* %[[i26]]
; CHECK-NEXT:   %29 = mul i64 %[[i27]], %[[i28]]
; CHECK-NEXT:   %mallocsize2 = mul nuw nsw i64 %29, 8
; CHECK-NEXT:   %malloccall3 = tail call noalias nonnull i8* @malloc(i64 %mallocsize2)
; CHECK-NEXT:   %cache.B = bitcast i8* %malloccall3 to double*
; CHECK-NEXT:   %30 = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %31 = load i64, i64* %30
; CHECK-NEXT:   %32 = bitcast i8* %B to double*
; CHECK:   %[[mul_i8:.+]] = add nuw nsw i64 %[[i27]], %[[i28]]
; CHECK-NEXT:   %33 = icmp eq i64 %[[mul_i8]], 0
; CHECK-NEXT:   br i1 %33, label %[[enzyme_memcpy_double_mat_64_exit21:.+]], label %[[init_idx:.+]]

; CHECK: [[init_idx]]:                                      ; preds = %[[init_end_i18:.+]], %__enzyme_memcpy_double_mat_64.exit
; CHECK-NEXT:   %[[j_i9:.+]] = phi i64 [ 0, %__enzyme_memcpy_double_mat_64.exit ], [ %[[j_next_i17:.+]], %[[init_end_i18]] ]
; CHECK-NEXT:   br label %[[for_body_i16:.+]]

; CHECK: [[for_body_i16]]:                                     ; preds = %[[for_body_i16]], %[[init_idx]]
; CHECK-NEXT:   %[[i_i11:.+]] = phi i64 [ 0, %[[init_idx]] ], [ %[[i_next_i15:.+]], %[[for_body_i16]] ]
; CHECK-NEXT:   %34 = mul nuw nsw i64 %[[j_i9]], %[[i27]]
; CHECK-NEXT:   %35 = add nuw nsw i64 %[[i_i11]], %34
; CHECK-NEXT:   %[[dst_i_i12:.+]] = getelementptr inbounds double, double* %cache.B, i64 %35
; CHECK-NEXT:   %36 = mul nuw nsw i64 %[[j_i9]], %31
; CHECK-NEXT:   %37 = add nuw nsw i64 %[[i_i11]], %36
; CHECK-NEXT:   %[[dst_i1_i13:.+]] = getelementptr inbounds double, double* %32, i64 %37
; CHECK-NEXT:   %[[src_i_l_i14:.+]] = load double, double* %[[dst_i1_i13]]
; CHECK-NEXT:   store double %[[src_i_l_i14]], double* %[[dst_i_i12]]
; CHECK-NEXT:   %[[i_next_i15]] = add nuw nsw i64 %[[i_i11]], 1
; CHECK-NEXT:   %38 = icmp eq i64 %[[i_next_i15]], %[[i27]]
; CHECK-NEXT:   br i1 %38, label %[[init_end_i18]], label %[[for_body_i16]]

; CHECK: [[init_end_i18]]:                                     
; CHECK-NEXT:   %[[j_next_i17]] = add nuw nsw i64 %[[j_i9]], 1
; CHECK-NEXT:   %39 = icmp eq i64 %[[j_next_i17]], %[[i28]]
; CHECK-NEXT:   br i1 %39, label %[[enzyme_memcpy_double_mat_64_exit21]], label %[[init_idx]]

; CHECK: [[enzyme_memcpy_double_mat_64_exit21]]:             ; preds = %__enzyme_memcpy_double_mat_64.exit, %[[init_end_i18]]
; CHECK-NEXT:   %40 = insertvalue { double*, double* } undef, double* %cache.A, 0
; CHECK-NEXT:   %41 = insertvalue { double*, double* } %40, double* %cache.B, 1
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %[[enzyme_memcpy_double_mat_64_exit21]]
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %tape.ext.A = extractvalue { double*, double* } %41, 0
; CHECK-NEXT:   %42 = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   %tape.ext.B = extractvalue { double*, double* } %41, 1
; CHECK-NEXT:   %43 = bitcast double* %tape.ext.B to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 0
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[r8:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[r9:.+]] = select i1 %[[r8]], i8 116, i8 0
; CHECK-DAG:    %[[r10:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[r11:.+]] = select i1 %[[r10]], i8 84, i8 %[[r9]]
; CHECK-DAG:    %[[r12:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[r13:.+]] = select i1 %[[r12]], i8 110, i8 %[[r11]]
; CHECK-DAG:    %[[r14:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[r15:.+]] = select i1 %[[r14]], i8 78, i8 %[[r13]]
; CHECK-NEXT:   store i8 %[[r15]], i8* %byref.transpose.transb
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   %loaded.trans4 = load i8, i8* %transb
; CHECK-DAG:   %[[r18:.+]] = icmp eq i8 %loaded.trans4, 78
; CHECK-DAG:   %[[r19:.+]] = icmp eq i8 %loaded.trans4, 110
; CHECK-DAG:   %[[r20:.+]] = or i1 %[[r19]], %[[r18]]
; CHECK-DAG:   %[[r21:.+]] = select i1 %[[r20]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %43, i8* %[[r21]], i8* %beta_p, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %loaded.trans5 = load i8, i8* %transa
; CHECK-DAG:   %[[r22:.+]] = icmp eq i8 %loaded.trans5, 78
; CHECK-DAG:   %[[r23:.+]] = icmp eq i8 %loaded.trans5, 110
; CHECK-DAG:   %[[r24:.+]] = or i1 %[[r23]], %[[r22]]
; CHECK-DAG:   %[[r25:.+]] = select i1 %[[r24]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %transb, i8* %k_p, i8* %n_p, i8* %m_p, i8* %alpha_p, i8* %42, i8* %[[r25]], i8* %"C'", i8* %ldc_p, i8* %beta_p, i8* %"B'", i8* %ldb_p)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   %intcast.constant.int.0 = bitcast i64* %byref.constant.int.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %[[byrefconstantint4]]
; CHECK-NEXT:   %[[intcast07:.+]] = bitcast i64* %[[byrefconstantint4]] to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %[[byrefconstantint5]]
; CHECK-NEXT:   %[[intcast09:.+]] = bitcast i64* %[[byrefconstantint5]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i8* %intcast.constant.int.0, i8* %[[intcast07]], i8* %fpcast.constant.fp.1.0, i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i8* %[[intcast09]])
; CHECK-NEXT:   %[[ret1:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[ret1]])
; CHECK-NEXT:   %[[ret2:.+]] = bitcast double* %tape.ext.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[ret2]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
