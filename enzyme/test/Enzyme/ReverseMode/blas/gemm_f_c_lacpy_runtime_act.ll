;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1  -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S -enzyme-detect-readthrow=0 | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i64, i64) 

define void @f(i8* noalias %C, i8* noalias %A, i8* noalias %B, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %ldb = alloca i64, align 16
  %ldb_p = bitcast i64* %ldb to i8*
  %ldc = alloca i64, align 16
  %ldc_p = bitcast i64* %ldc to i8*
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p, i64 1, i64 1) 
  %ptr = bitcast i8* %A to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB, i8* %alpha, i8* %dalpha, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %A, metadata !"enzyme_dup", i8* %B, i8* %dB, metadata !"enzyme_dup", i8* %alpha, i8* %dalpha, metadata !"enzyme_dup", i8* %beta, i8* %beta)
  ret void
}

; CHECK: define internal void @diffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B, i8* %"B'", i8* noalias %alpha, i8* %"alpha'", i8* noalias %beta, i8*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[byref_one_i21:.+]] = alloca i64
; CHECK-NEXT:   %[[byref_mat_size_i24:.+]] = alloca i64
; CHECK-NEXT:   %byref.constant.one.i = alloca i64
; CHECK-NEXT:   %byref.mat.size.i = alloca i64
; CHECK-NEXT:   %[[byrefgarbage:.+]] = alloca i8
; CHECK-NEXT:   %[[byrefgarbage2:.+]] = alloca i8
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %[[fp109:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[byref_fp_1_017:.+]] = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %[[byref_int_019:.+]] = alloca i64, align 8
; CHECK-NEXT:   %[[byref_fp_021:.+]] = alloca double, align 8
; CHECK-NEXT:   %[[tmp:.+]] = alloca i64
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %ldb = alloca i64, align 16
; CHECK-NEXT:   %ldb_p = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %rt.tmp.inactive.alpha = icmp eq i8* %"alpha'", %alpha
; CHECK-NEXT:   %rt.tmp.inactive.A = icmp eq i8* %"A'", %A
; CHECK-NEXT:   %rt.tmp.inactive.B = icmp eq i8* %"B'", %B
; CHECK-NEXT:   %rt.tmp.inactive.beta = icmp eq i8* %"beta'", %beta
; CHECK-NEXT:   %rt.tmp.inactive.C = icmp eq i8* %"C'", %C
; CHECK-NEXT:   %rt.inactive.alpha = or i1 %rt.tmp.inactive.alpha, %rt.tmp.inactive.C
; CHECK-NEXT:   %rt.inactive.A = or i1 %rt.tmp.inactive.A, %rt.tmp.inactive.C
; CHECK-NEXT:   %rt.inactive.B = or i1 %rt.tmp.inactive.B, %rt.tmp.inactive.C
; CHECK-NEXT:   %rt.inactive.beta = or i1 %rt.tmp.inactive.beta, %rt.tmp.inactive.C
; CHECK-NEXT:   %rt.inactive.C = or i1 %rt.tmp.inactive.C, %rt.tmp.inactive.C
; CHECK-NEXT:   %loaded.trans = load i8, i8* %transa
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[i2:.+]] = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = select i1 %[[i2]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[i4:.+]] = select i1 %[[i2]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %[[i3]] to i64*
; CHECK-NEXT:   %[[i6:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = bitcast i8* %[[i4]] to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = mul i64 %[[i6]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[i9]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %[[byrefgarbage]]
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %[[i3]], i8* %[[i4]], i8* %A, i8* %lda_p, double* %cache.A, i8* %[[i3]], i64 1)
; CHECK-NEXT:   %[[i10:.+]] = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %[[i11:.+]] = load i64, i64* %[[i10]]
; CHECK-NEXT:   %[[i12:.+]] = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %[[i13:.+]] = load i64, i64* %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = mul i64 %[[i11]], %[[i13]]
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %[[i14]], 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %cache.C = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage3
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage3, i8* %m_p, i8* %n_p, i8* %C, i8* %ldc_p, double* %cache.C, i8* %m_p, i64 1)
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry

; CHECK-NEXT: %[[pcmp:.+]] = icmp ne double* %ptr, %"ptr'ipc"
; CHECK-NEXT: br i1 %[[pcmp]], label %invertentry_active, label %invertentry_amerge


; CHECK: invertentry_active:  
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   br label %invertentry_amerge

; CHECK: invertentry_amerge:
; CHECK-NEXT:   %[[matA:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   %[[matC0:.+]] = bitcast double* %cache.C to i8*
; CHECK-NEXT:   %[[matC:.+]] = bitcast double* %cache.C to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   br i1 %rt.inactive.alpha, label %invertentry_amerge.alpha.done, label %invertentry_amerge.alpha.active

; CHECK: invertentry_amerge.alpha.active:                         ; preds = %invertentry
; CHECK-NEXT:   %[[i17:.+]] = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %[[i18:.+]] = load i64, i64* %[[i17]]
; CHECK-NEXT:   %[[i19:.+]] = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %[[i20:.+]] = load i64, i64* %[[i19]]
; CHECK-NEXT:   %size_AB = mul nuw i64 %[[i18]], %[[i20]]
; CHECK-NEXT:   %mallocsize5 = mul nuw nsw i64 %size_AB, 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize5)
; CHECK-NEXT:   %[[matAB:.+]] = bitcast i8* %malloccall6 to double*
; CHECK-NEXT:   %[[i21:.+]] = bitcast double* %[[matAB:.+]] to i8*

; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   %loaded.trans7 = load i8, i8* %transa
; CHECK-DAG:   %[[i41:.+]] = icmp eq i8 %loaded.trans7, 78
; CHECK-DAG:   %[[i42:.+]] = icmp eq i8 %loaded.trans7, 110
; CHECK-NEXT:   %[[i43:.+]] = or i1 %[[i42]], %[[i41]]
; CHECK-NEXT:   %[[i44:.+]] = select i1 %[[i43]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %fpcast.constant.fp.1.0, i8* %[[matA]], i8* %[[i44]], i8* %B, i8* %ldb_p, i8* %fpcast.constant.fp.0.0, i8* %[[i21]], i8* %m_p, i64 1, i64 1)
; CHECK:   %[[i45:.+]] = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %[[i46:.+]] = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   store i64 1, i64* %byref.constant.one.i
; CHECK-NEXT:   %intcast.constant.one.i = bitcast i64* %byref.constant.one.i to i8*
; CHECK-DAG:   %[[i47:.+]] = load i64, i64* %m
; CHECK-DAG:   %[[i48:.+]] = load i64, i64* %n
; CHECK-DAG:   %mat.size.i = mul nuw i64 %[[i47]], %[[i48]]
; CHECK-NEXT:   store i64 %mat.size.i, i64* %byref.mat.size.i
; CHECK-NEXT:   %intcast.mat.size.i = bitcast i64* %byref.mat.size.i to i8*
; CHECK-NEXT:   %[[i49:.+]] = icmp eq i64 %mat.size.i, 0
; CHECK-NEXT:   br i1 %[[i49]], label %__enzyme_inner_prodd_64_.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %invertentry_amerge.alpha.active
; CHECK-NEXT:   %[[i50:.+]] = load i64, i64* %ldc
; CHECK-NEXT:   %[[i51:.+]] = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %[[i52:.+]] = icmp eq i64 %[[i47]], %[[i50]]
; CHECK-NEXT:   br i1 %[[i52]], label %fast.path.i, label %for.body.i

; CHECK: fast.path.i:                                      ; preds = %init.idx.i
; CHECK-NEXT:   %[[i53:.+]] = call fast double @ddot_64_(i8* %intcast.mat.size.i, i8* %"C'", i8* %intcast.constant.one.i, i8* %[[i21]], i8* %intcast.constant.one.i)
; CHECK-NEXT:   br label %__enzyme_inner_prodd_64_.exit

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %Aidx.i = phi i64 [ 0, %init.idx.i ], [ %Aidx.next.i, %for.body.i ]
; CHECK-NEXT:   %Bidx.i = phi i64 [ 0, %init.idx.i ], [ %Bidx.next.i, %for.body.i ]
; CHECK-NEXT:   %iteration.i = phi i64 [ 0, %init.idx.i ], [ %iter.next.i, %for.body.i ]
; CHECK-NEXT:   %sum.i = phi{{( fast)?}} double [ 0.000000e+00, %init.idx.i ], [ %[[i57:.+]], %for.body.i ]
; CHECK-NEXT:   %A.i.i = getelementptr inbounds double, double* %[[i51]], i64 %Aidx.i
; CHECK-NEXT:   %B.i.i = getelementptr inbounds double, double* %[[matAB]], i64 %Bidx.i
; CHECK-NEXT:   %[[i54:.+]] = bitcast double* %A.i.i to i8*
; CHECK-NEXT:   %[[i55:.+]] = bitcast double* %B.i.i to i8*
; CHECK-NEXT:   %[[i56:.+]] = call fast double @ddot_64_(i8* %m_p, i8* %[[i54]], i8* %intcast.constant.one.i, i8* %[[i55]], i8* %intcast.constant.one.i)
; CHECK-NEXT:   %Aidx.next.i = add nuw i64 %Aidx.i, %[[i50]]
; CHECK-NEXT:   %Bidx.next.i = add nuw i64 %Aidx.i, %[[i47]]
; CHECK-NEXT:   %iter.next.i = add i64 %iteration.i, 1
; CHECK-NEXT:   %[[i57]] = fadd fast double %sum.i, %[[i56]]
; CHECK-NEXT:   %[[i58:.+]] = icmp eq i64 %iteration.i, %[[i48]]
; CHECK-NEXT:   br i1 %[[i58]], label %__enzyme_inner_prodd_64_.exit, label %for.body.i

; CHECK: __enzyme_inner_prodd_64_.exit:                    ; preds = %invertentry_amerge.alpha.active, %fast.path.i, %for.body.i
; CHECK-NEXT:   %res.i = phi double [ 0.000000e+00, %invertentry_amerge.alpha.active ], [ %sum.i, %for.body.i ], [ %[[i53]], %fast.path.i ]
; CHECK-NEXT:   %[[i59:.+]] = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %[[i60:.+]] = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   %[[i61:.+]] = bitcast i8* %"alpha'" to double*
; CHECK-NEXT:   %[[i62:.+]] = load double, double* %[[i61]]
; CHECK-NEXT:   %[[i63:.+]] = fadd fast double %[[i62]], %res.i
; CHECK-NEXT:   store double %[[i63]], double* %[[i61]]
; CHECK-NEXT:  %[[forfree:.+]] = bitcast double* %[[matAB]] to i8* 
; CHECK-NEXT:  tail call void @free(i8* nonnull %[[forfree:.+]]) 
; CHECK-NEXT:   br label %invertentry_amerge.alpha.done

; CHECK: invertentry_amerge.alpha.done:                           ; preds = %__enzyme_inner_prodd_64_.exit, %invertentry
; CHECK-NEXT:   br i1 %rt.inactive.A, label %invertentry_amerge.A.done, label %invertentry_amerge.A.active

; CHECK: invertentry_amerge.A.active:                             ; preds = %invertentry_amerge.alpha.done
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[i33:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[i34:.+]] = select i1 %[[i33]], i8 116, i8 78
; CHECK-DAG:    %[[i35:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[i36:.+]] = select i1 %[[i35]], i8 84, i8 %[[i34]]
; CHECK-DAG:    %[[i37:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[i38:.+]] = select i1 %[[i37]], i8 110, i8 %[[i36]]
; CHECK-DAG:    %[[i39:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[i40:.+]] = select i1 %[[i39]], i8 78, i8 %[[i38]]
; CHECK-NEXT:   store i8 %[[i40]], i8* %byref.transpose.transb
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r62:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[r63:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r64:.+]] = or i1 %[[r63]], %[[r62]]
; CHECK-NEXT:   %[[r65:.+]] = select i1 %[[r64]], i8* %transa, i8* %transb
; CHECK-NEXT:   %[[r66:.+]] = select i1 %[[r64]], i8* %byref.transpose.transb, i8* %transa
; CHECK-NEXT:   %[[r67:.+]] = select i1 %[[r64]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[r68:.+]] = select i1 %[[r64]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %ld.row.trans8 = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r69:.+]] = icmp eq i8 %ld.row.trans8, 110
; CHECK-NEXT:   %[[r70:.+]] = icmp eq i8 %ld.row.trans8, 78
; CHECK-NEXT:   %[[r71:.+]] = or i1 %[[r70]], %[[r69]]
; CHECK-NEXT:   %[[r72:.+]] = select i1 %[[r71]], i8* %"C'", i8* %B
; CHECK-NEXT:   %[[r73:.+]] = select i1 %[[r71]], i8* %ldc_p, i8* %ldb_p
; CHECK-NEXT:   %[[r74:.+]] = select i1 %[[r71]], i8* %B, i8* %"C'"
; CHECK-NEXT:   %[[r75:.+]] = select i1 %[[r71]], i8* %ldb_p, i8* %ldc_p
; CHECK-NEXT:   store double 1.000000e+00, double* %[[fp109]], align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.010 = bitcast double* %[[fp109]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r65]], i8* %[[r66]], i8* %[[r67]], i8* %[[r68]], i8* %n_p, i8* %alpha, i8* %[[r72]], i8* %[[r73]], i8* %[[r74]], i8* %[[r75]], i8* %fpcast.constant.fp.1.010, i8* %"A'", i8* %lda_p, i64 1, i64 1)
; CHECK-NEXT:   br label %invertentry_amerge.A.done

; CHECK: invertentry_amerge.A.done:                               ; preds = %invertentry_amerge.A.active, %invertentry_amerge.alpha.done
; CHECK-NEXT:   br i1 %rt.inactive.B, label %invertentry_amerge.B.done, label %invertentry_amerge.B.active

; CHECK: invertentry_amerge.B.active:                             ; preds = %invertentry_amerge.A.done
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[i25:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[i26:.+]] = select i1 %[[i25]], i8 116, i8 78
; CHECK-DAG:    %[[i27:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[i28:.+]] = select i1 %[[i27]], i8 84, i8 %[[i26]]
; CHECK-DAG:    %[[i29:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[i30:.+]] = select i1 %[[i29]], i8 110, i8 %[[i28]]
; CHECK-DAG:    %[[i31:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[i32:.+]] = select i1 %[[i31]], i8 78, i8 %[[i30]]
; CHECK-NEXT:   store i8 %[[i32]], i8* %byref.transpose.transa
; CHECK-NEXT:   %[[ld_row_trans13:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r76:.+]] = icmp eq i8 %[[ld_row_trans13]], 110
; CHECK-NEXT:   %[[r77:.+]] = icmp eq i8 %[[ld_row_trans13]], 78
; CHECK-NEXT:   %[[r78:.+]] = or i1 %[[r77]], %[[r76]]
; CHECK-NEXT:   %[[r79:.+]] = select i1 %[[r78]], i8* %byref.transpose.transa, i8* %transb
; CHECK-NEXT:   %[[r80:.+]] = select i1 %[[r78]], i8* %transb, i8* %transa
; CHECK-NEXT:   %[[r81:.+]] = select i1 %[[r78]], i8* %k_p, i8* %n_p
; CHECK-NEXT:   %[[r82:.+]] = select i1 %[[r78]], i8* %n_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans14:.+]] = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r83:.+]] = icmp eq i8 %[[loaded_trans14]], 78
; CHECK-NEXT:   %[[r84:.+]] = icmp eq i8 %[[loaded_trans14]], 110
; CHECK-NEXT:   %[[r85:.+]] = or i1 %[[r84]], %[[r83]]
; CHECK-NEXT:   %[[r86:.+]] = select i1 %[[r85]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[loaded_trans15:.+]] = load i8, i8* %transa, align 1
; CHECK-NEXT:   %[[r87:.+]] = icmp eq i8 %[[loaded_trans15]], 78
; CHECK-NEXT:   %[[r88:.+]] = icmp eq i8 %[[loaded_trans15]], 110
; CHECK-NEXT:   %[[r89:.+]] = or i1 %[[r88]], %[[r87]]
; CHECK-NEXT:   %[[r90:.+]] = select i1 %[[r89]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[ld_row_trans16:.+]] = load i8, i8* %transb, align 1
; CHECK-NEXT:   %[[r91:.+]] = icmp eq i8 %[[ld_row_trans16]], 110
; CHECK-NEXT:   %[[r92:.+]] = icmp eq i8 %[[ld_row_trans16]], 78
; CHECK-NEXT:   %[[r93:.+]] = or i1 %[[r92]], %[[r91]]
; CHECK-NEXT:   %[[r94:.+]] = select i1 %[[r93]], i8* %[[matA]], i8* %"C'"
; CHECK-NEXT:   %[[r95:.+]] = select i1 %[[r93]], i8* %[[r90]], i8* %ldc_p
; CHECK-NEXT:   %[[r96:.+]] = select i1 %[[r93]], i8* %"C'", i8* %[[matA]]
; CHECK-NEXT:   %[[r97:.+]] = select i1 %[[r93]], i8* %ldc_p, i8* %[[r86]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_1_017]], align 8
; CHECK-NEXT:   %[[fpcast_1_018:.+]] = bitcast double* %[[byref_fp_1_017]] to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r79]], i8* %[[r80]], i8* %[[r81]], i8* %[[r82]], i8* %m_p, i8* %alpha, i8* %[[r94]], i8* %[[r95]], i8* %[[r96]], i8* %[[r97]], i8* %[[fpcast_1_018]], i8* %"B'", i8* %ldb_p, i64 1, i64 1)
; CHECK-NEXT:   br label %invertentry_amerge.B.done

; CHECK: invertentry_amerge.B.done:                               ; preds = %invertentry_amerge.B.active, %invertentry_amerge.A.done
; CHECK-NEXT:   br i1 %rt.inactive.beta, label %invertentry_amerge.beta.done, label %invertentry_amerge.beta.active

; CHECK: invertentry_amerge.beta.active:                          ; preds = %invertentry_amerge.B.done
; CHECK:   %[[i68:.+]] = bitcast i64* %[[byrefconstantonei15:.+]] to i8*
; CHECK:   %[[i69:.+]] = bitcast i64* %[[byrefmatsizei18:.+]] to i8*
; CHECK:   store i64 1, i64* %[[byref_one_i21]]
; CHECK-NEXT:   %[[intcast_one_i24:.+]] = bitcast i64* %[[byref_one_i21]] to i8*
; CHECK-NEXT:   %[[i70:.+]] = load i64, i64* %m
; CHECK-NEXT:   %[[i71:.+]] = load i64, i64* %n
; CHECK-NEXT:   %[[mat_size_i25:.+]] = mul nuw i64 %[[i70]], %[[i71]]
; CHECK-NEXT:   store i64 %[[mat_size_i25]], i64* %[[byrefmatsizei18]]
; CHECK-NEXT:   %[[intcast_mat_size_i27:.+]] = bitcast i64* %[[byref_mat_size_i24]] to i8*
; CHECK-NEXT:   %[[i72:.+]] = icmp eq i64 %[[mat_size_i25]], 0
; CHECK-NEXT:   br i1 %[[i72]], label %[[__enzyme_inner_prodd_64_exit41:.+]], label %[[init_idx_i28:.+]]

; CHECK: [[init_idx_i28]]:                                     ; preds = %invertentry_amerge.beta.active
; CHECK-NEXT:   %[[i73:.+]] = load i64, i64* %ldc
; CHECK-NEXT:   %[[i74:.+]] = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %[[i75:.+]] = icmp eq i64 %[[i70]], %[[i73]]
; CHECK-NEXT:   br i1 %[[i75]], label %[[fast_path_i29:.+]], label %[[for_body_i39:.+]]

; CHECK: [[fast_path_i29]]:                                    ; preds = %[[init_idx_i28]]
; CHECK-NEXT:   %[[i76:.+]] = call fast double @ddot_64_(i8* %[[intcast_mat_size_i27]], i8* %"C'", i8* %[[intcast_one_i24]], i8* %[[matC0]], i8* %[[intcast_one_i24]])
; CHECK-NEXT:   br label %[[__enzyme_inner_prodd_64_exit41]]

; CHECK: [[forbodyi31:.+]]:                                     ; preds = %[[forbodyi31]], %[[initidxi20:.+]]
; CHECK-NEXT:   %[[Aidxi22:.+]] = phi i64 [ 0, %[[initidxi20]] ], [ %[[Aidxnexti28:.+]], %[[forbodyi31]] ]
; CHECK-NEXT:   %[[Bidxi23:.+]] = phi i64 [ 0, %[[initidxi20]] ], [ %[[Bidxnexti29:.+]], %[[forbodyi31]] ]
; CHECK-NEXT:   %[[iterationi24:.+]] = phi i64 [ 0, %[[initidxi20]] ], [ %[[iternexti30:.+]], %[[forbodyi31]] ]
; CHECK-NEXT:   %[[sumi25:.+]] = phi fast double [ 0.000000e+00, %[[initidxi20]] ], [ %[[i80:.+]], %[[forbodyi31]] ]
; CHECK-NEXT:   %[[Aii26:.+]] = getelementptr inbounds double, double* %[[i74]], i64 %[[Aidxi22]]
; CHECK-NEXT:   %[[Bii27:.+]] = getelementptr inbounds double, double* %cache.C, i64 %[[Bidxi23]]
; CHECK-NEXT:   %[[i77:.+]] = bitcast double* %[[Aii26]] to i8*
; CHECK-NEXT:   %[[i78:.+]] = bitcast double* %[[Bii27]] to i8*
; CHECK-NEXT:   %[[i79:.+]] = call fast double @ddot_64_(i8* %m_p, i8* %[[i77]], i8* %[[intcast_one_i24]], i8* %[[i78]], i8* %[[intcast_one_i24]])
; CHECK-NEXT:   %[[Aidxnexti28]] = add nuw i64 %[[Aidxi22]], %[[i73]]
; CHECK-NEXT:   %[[Bidxnexti29]] = add nuw i64 %[[Aidxi22]], %[[i70]]
; CHECK-NEXT:   %[[iternexti30]] = add i64 %[[iterationi24]], 1
; CHECK-NEXT:   %[[i80]] = fadd fast double %[[sumi25]], %[[i79]]
; CHECK-NEXT:   %[[i81:.+]] = icmp eq i64 %[[iterationi24]], %[[i71]]
; CHECK-NEXT:   br i1 %[[i81]], label %[[__enzyme_inner_prodd_64_exit41]], label %[[forbodyi31]]

; CHECK: [[__enzyme_inner_prodd_64_exit41]]:                  ; preds = %invertentry_amerge.beta.active, %[[fast_path_i29]], %[[for_body_i39]]
; CHECK-NEXT:   %[[resi32:.+]] = phi double [ 0.000000e+00, %invertentry_amerge.beta.active ], [ %[[sumi25]], %[[forbodyi31]] ], [ %[[i76]], %[[fast_path_i29]] ]
; CHECK-NEXT:   %[[i82:.+]] = bitcast i64* %[[byrefconstantonei15]] to i8*
; CHECK:   %[[i83:.+]] = bitcast i64* %[[byrefmatsizei18]] to i8*
; CHECK:   %[[i84:.+]] = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %[[i85:.+]] = load double, double* %[[i84]]
; CHECK-NEXT:   %[[i86:.+]] = fadd fast double %[[i85]], %[[resi32]]
; CHECK-NEXT:   store double %[[i86]], double* %[[i84]]
; CHECK-NEXT:   br label %invertentry_amerge.beta.done

; CHECK: invertentry_amerge.beta.done:                            ; preds = %[[__enzyme_inner_prodd_64_exit41]], %invertentry_amerge.B.done
; CHECK-NEXT:   br i1 %rt.inactive.C, label %invertentry_amerge.C.done, label %invertentry_amerge.C.active

; CHECK: invertentry_amerge.C.active:                             ; preds = %invertentry_amerge.beta.done
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[byref_int_019]]
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byref_fp_021]]
; CHECK-NEXT:   %[[fpcast_1_022:.+]] = bitcast double* %[[byref_fp_021]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byref_int_019]], i8* %[[fpcast_1_022]], i8* %beta, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   br label %invertentry_amerge.C.done

; CHECK: invertentry_amerge.C.done:                               ; preds = %invertentry_amerge.C.active, %invertentry_amerge.beta.done
; CHECK-NEXT:   %[[i87:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i87]])
; CHECK-NEXT:   %[[i88:.+]] = bitcast double* %cache.C to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i88]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
