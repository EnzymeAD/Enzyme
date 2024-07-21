;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i64, i64) 

declare i8* @AData(i64)
declare i8* @Aldap(i64)
declare i8* @malloc(i64)
declare void @free(i8*)

define void @f(i8* %C, i8* %B) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %n = alloca i64, align 16
  store i64 4, i64* %n, align 16
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
  %alpha = alloca double, align 16
  %alpha_p = bitcast double* %alpha to i8*
  %ldb = alloca i64, align 16
  %ldb_p = bitcast i64* %ldb to i8*
  %beta = alloca double, align 16
  %beta_p = bitcast double* %beta to i8*
  %ldc = alloca i64, align 16
  %ldc_p = bitcast i64* %ldc to i8*
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  %m_p = call i8* @malloc(i64 8)
  %m = bitcast i8* %m_p to i64*
  store i64 4, i64* %m, align 16
  store i64 8, i64* %k, align 16
  %n_p = bitcast i64* %n to i8*
  store double 1.000000e+00, double* %alpha, align 16
  store i64 8, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  %A = call i8* @AData(i64 %i) "enzyme_inactive"
  %lda_p = call i8* @Aldap(i64 %i) "enzyme_inactive"
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1) 
  call void @free(i8* %m_p)
  %cmp = icmp eq i64 %inc, 10
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %B, i8* %dB) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %B, i8* %dB)
  ret void
}

; CHECK: define internal void @diffef(i8* %C, i8* %"C'", i8* %B, i8* %"B'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"iv'ac" = alloca i64, align 8
; CHECK-NEXT:   %malloccall_cache = alloca i8**, align 8
; CHECK-NEXT:   %lda_p_cache = alloca i8**, align 8
; CHECK-NEXT:   %byref.transa = alloca i8
; CHECK-NEXT:   %byref.transb = alloca i8
; CHECK-NEXT:   %byref.k = alloca i64
; CHECK-NEXT:   %byref.alpha = alloca double
; CHECK-NEXT:   %byref.lda = alloca i64
; CHECK-NEXT:   %byref.ldb = alloca i64
; CHECK-NEXT:   %byref.beta = alloca double
; CHECK-NEXT:   %byref.ldc = alloca i64
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %m_p_cache = alloca i8*, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.033 = alloca i64, align 8
; CHECK-NEXT:   %[[fp035:.+]] = alloca double, align 8
; CHECK-NEXT:   %[[tmp:.+]] = alloca i64
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %alpha = alloca double, align 16
; CHECK-NEXT:   %alpha_p = bitcast double* %alpha to i8*
; CHECK-NEXT:   %ldb = alloca i64, align 16
; CHECK-NEXT:   %ldb_p = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %beta = alloca double, align 16
; CHECK-NEXT:   %beta_p = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   %[[malloccall4:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80), !enzyme_cache_alloc !0
; CHECK-NEXT:   %malloccall_malloccache = bitcast i8* %[[malloccall4]] to i8**
; CHECK-NEXT:   store i8** %malloccall_malloccache, i8*** %malloccall_cache, align 8, !invariant.group !2
; CHECK-NEXT:   %[[malloccall14:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80), !enzyme_cache_alloc !3
; CHECK-NEXT:   %lda_p_malloccache = bitcast i8* %[[malloccall14]] to i8**
; CHECK-NEXT:   store i8** %lda_p_malloccache, i8*** %lda_p_cache, align 8, !invariant.group !5
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %__enzyme_memcpy_double_mat_64.exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %__enzyme_memcpy_double_mat_64.exit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   %m_p = call i8* @malloc(i64 8)
; CHECK-NEXT:   %m = bitcast i8* %m_p to i64*
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i8* %m_p, i8** %m_p_cache, align 8, !invariant.group !6
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %A = call i8* @AData(i64 %iv) #
; CHECK-NEXT:   %lda_p = call i8* @Aldap(i64 %iv) #
; CHECK-NEXT:   %avld.transa = load i8, i8* %transa
; CHECK-NEXT:   %avld.transb = load i8, i8* %transb
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
; CHECK-NEXT:   %loaded.trans = load i8, i8* %transa
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[z2:.+]] = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[z3:.+]] = select i1 %[[z2]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   %[[z4:.+]] = select i1 %[[z2]], i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %[[z3]] to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %[[z4]] to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %[[z9:.+]] = mul i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[z9]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[z10:.+]] = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !15, !invariant.group !2
; CHECK-NEXT:   %[[z11:.+]] = getelementptr inbounds i8*, i8** %[[z10]], i64 %iv
; CHECK-NEXT:   store i8* %malloccall, i8** %[[z11]], align 8, !invariant.group !16
; CHECK-NEXT:   %[[z12:.+]] = load i8**, i8*** %lda_p_cache, align 8, !dereferenceable !15, !invariant.group !5
; CHECK-NEXT:   %[[z13:.+]] = getelementptr inbounds i8*, i8** %[[z12]], i64 %iv
; CHECK-NEXT:   store i8* %lda_p, i8** %[[z13]], align 8, !invariant.group !17
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %[[z14:.+]] = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %[[z15:.+]] = load i64, i64* %[[z14]]
; CHECK-NEXT:   %[[z16:.+]] = bitcast i8* %A to double*
; CHECK:   %mul.i = add nuw nsw i64 %[[i7]], %[[i8]]
; CHECK-NEXT:   %[[z17:.+]] = icmp eq i64 %mul.i, 0
; CHECK-NEXT:   br i1 %[[z17]], label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %init.end.i, %loop
; CHECK-NEXT:   %j.i = phi i64 [ 0, %loop ], [ %j.next.i, %init.end.i ]
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %i.i = phi i64 [ 0, %init.idx.i ], [ %i.next.i, %for.body.i ]
; CHECK-NEXT:   %[[z18:.+]] = mul nuw nsw i64 %j.i, %[[i7]]
; CHECK-NEXT:   %[[z19:.+]] = add nuw nsw i64 %i.i, %[[z18]]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %cache.A, i64 %[[z19]]
; CHECK-NEXT:   %[[z20:.+]] = mul nuw nsw i64 %j.i, %[[z15]]
; CHECK-NEXT:   %[[z21:.+]] = add nuw nsw i64 %i.i, %[[z20]]
; CHECK-NEXT:   %dst.i1.i = getelementptr inbounds double, double* %[[z16]], i64 %[[z21]]
; CHECK-NEXT:   %src.i.l.i = load double, double* %dst.i1.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %i.next.i = add nuw nsw i64 %i.i, 1
; CHECK-NEXT:   %[[z22:.+]] = icmp eq i64 %i.next.i, %[[i7]]
; CHECK-NEXT:   br i1 %[[z22]], label %init.end.i, label %for.body.i

; CHECK: init.end.i:                                       ; preds = %for.body.i
; CHECK-NEXT:   %j.next.i = add nuw nsw i64 %j.i, 1
; CHECK-NEXT:   %[[z23:.+]] = icmp eq i64 %j.next.i, %[[i8]]
; CHECK-NEXT:   br i1 %[[z23]], label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: __enzyme_memcpy_double_mat_64.exit:               ; preds = %loop, %init.end.i
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p, i64 1, i64 1)
; CHECK-NEXT:   call void @free(i8* %m_p)
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp, label %exit, label %loop

; CHECK: exit:                                             ; preds = %__enzyme_memcpy_double_mat_64.exit
; CHECK-NEXT:   br label %invertexit

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %[[i33:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %forfree = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !15, !invariant.group !2
; CHECK-NEXT:   %[[i34:.+]] = bitcast i8** %forfree to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i34]]), !enzyme_cache_free !0
; CHECK-NEXT:   %[[i35:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[forfree15:.+]] = load i8**, i8*** %lda_p_cache, align 8, !dereferenceable !15, !invariant.group !5
; CHECK-NEXT:   %[[i36:.+]] = bitcast i8** %[[forfree15]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i36]]), !enzyme_cache_free !3
; CHECK-NEXT:   ret void

; CHECK: invertloop:                                       ; preds = %remat_loop_loop, %remat_loop_loop
; CHECK-NEXT:   %[[i37:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[i38:.+]] = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !15, !invariant.group !2
; CHECK-NEXT:   %[[i39:.+]] = getelementptr inbounds i8*, i8** %[[i38]], i64 %[[i37]]
; CHECK-NEXT:   %[[i40:.+]] = load i8*, i8** %[[i39]], align 8, !invariant.group !16
; CHECK-NEXT:   %cache.A_unwrap = bitcast i8* %[[i40]] to double*
; CHECK-NEXT:   %pcld.ldc_unwrap = bitcast i8* %ldc_p to i64*
; CHECK-NEXT:   %avld.ldc_unwrap = load i64, i64* %pcld.ldc_unwrap
; CHECK-NEXT:   %pcld.beta_unwrap = bitcast i8* %beta_p to double*
; CHECK-NEXT:   %avld.beta_unwrap = load double, double* %pcld.beta_unwrap
; CHECK-NEXT:   %pcld.ldb_unwrap = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %avld.ldb_unwrap = load i64, i64* %pcld.ldb_unwrap
; CHECK-NEXT:   %[[i41:.+]] = load i8**, i8*** %lda_p_cache, align 8, !dereferenceable !15, !invariant.group !5
; CHECK-NEXT:   %[[i42:.+]] = getelementptr inbounds i8*, i8** %[[i41]], i64 %[[i37]]
; CHECK-NEXT:   %[[i43:.+]] = load i8*, i8** %[[i42]], align 8, !invariant.group !17
; CHECK-NEXT:   %pcld.lda_unwrap = bitcast i8* %[[i43]] to i64*
; CHECK-NEXT:   %avld.lda_unwrap = load i64, i64* %pcld.lda_unwrap
; CHECK-NEXT:   %pcld.alpha_unwrap = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %avld.alpha_unwrap = load double, double* %pcld.alpha_unwrap
; CHECK-NEXT:   %pcld.k_unwrap = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %avld.k_unwrap = load i64, i64* %pcld.k_unwrap
; CHECK-NEXT:   %avld.transb_unwrap = load i8, i8* %transb
; CHECK-NEXT:   %avld.transa_unwrap = load i8, i8* %transa
; CHECK-NEXT:   store i8 %avld.transa_unwrap, i8* %byref.transa
; CHECK-NEXT:   store i8 %avld.transb_unwrap, i8* %byref.transb
; CHECK-NEXT:   store i64 %avld.k_unwrap, i64* %byref.k
; CHECK-NEXT:   %cast.k = bitcast i64* %byref.k to i8*
; CHECK-NEXT:   store double %avld.alpha_unwrap, double* %byref.alpha
; CHECK-NEXT:   %cast.alpha = bitcast double* %byref.alpha to i8*
; CHECK-NEXT:   store i64 %avld.lda_unwrap, i64* %byref.lda
; CHECK-NEXT:   %cast.lda = bitcast i64* %byref.lda to i8*
; CHECK-NEXT:   store i64 %avld.ldb_unwrap, i64* %byref.ldb
; CHECK-NEXT:   %cast.ldb = bitcast i64* %byref.ldb to i8*
; CHECK-NEXT:   store double %avld.beta_unwrap, double* %byref.beta
; CHECK-NEXT:   %cast.beta = bitcast double* %byref.beta to i8*
; CHECK-NEXT:   store i64 %avld.ldc_unwrap, i64* %byref.ldc
; CHECK-NEXT:   %cast.ldc = bitcast i64* %byref.ldc to i8*
; CHECK-NEXT:   %[[r35:.+]] = bitcast double* %cache.A_unwrap to i8*
; CHECK-NEXT:   %[[r36:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[r37:.+]] = load i8*, i8** %m_p_cache, align 8, !invariant.group !6
; CHECK-NEXT:   %[[r38:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %n_p_unwrap = bitcast i64* %n to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %ld.transa = load i8, i8* %byref.transa
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 78
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %byref.transb, align 1
; CHECK-NEXT:   %[[r55:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[r56:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r57:.+]] = or i1 %[[r56]], %[[r55]]
; CHECK-NEXT:   %[[r58:.+]] = select i1 %[[r57]], i8* %byref.transpose.transa, i8* %byref.transb
; CHECK-NEXT:   %[[r59:.+]] = select i1 %[[r57]], i8* %byref.transb, i8* %byref.transa
; CHECK-NEXT:   %[[r60:.+]] = select i1 %[[r57]], i8* %cast.k, i8* %n_p_unwrap
; CHECK-NEXT:   %[[r61:.+]] = select i1 %[[r57]], i8* %n_p_unwrap, i8* %cast.k
; CHECK-NEXT:   %loaded.trans30 = load i8, i8* %byref.transa, align 1
; CHECK-NEXT:   %[[r62:.+]] = icmp eq i8 %loaded.trans30, 78
; CHECK-NEXT:   %[[r63:.+]] = icmp eq i8 %loaded.trans30, 110
; CHECK-NEXT:   %[[r64:.+]] = or i1 %[[r63]], %[[r62]]
; CHECK-NEXT:   %[[r65:.+]] = select i1 %[[r64]], i8* %[[r37]], i8* %cast.k
; CHECK-NEXT:   %loaded.trans31 = load i8, i8* %byref.transa, align 1
; CHECK-NEXT:   %[[r66:.+]] = icmp eq i8 %loaded.trans31, 78
; CHECK-NEXT:   %[[r67:.+]] = icmp eq i8 %loaded.trans31, 110
; CHECK-NEXT:   %[[r68:.+]] = or i1 %[[r67]], %[[r66]]
; CHECK-NEXT:   %[[r69:.+]] = select i1 %[[r68]], i8* %[[r37]], i8* %cast.k 
; CHECK-NEXT:   %ld.row.trans32 = load i8, i8* %byref.transb, align 1
; CHECK-NEXT:   %[[r70:.+]] = icmp eq i8 %ld.row.trans32, 110
; CHECK-NEXT:   %[[r71:.+]] = icmp eq i8 %ld.row.trans32, 78
; CHECK-NEXT:   %[[r72:.+]] = or i1 %[[r71]], %[[r70]]
; CHECK-NEXT:   %[[r73:.+]] = select i1 %[[r72]], i8* %[[r35]], i8* %"C'"
; CHECK-NEXT:   %[[r74:.+]] = select i1 %[[r72]], i8* %[[r69]], i8* %cast.ldc
; CHECK-NEXT:   %[[r75:.+]] = select i1 %[[r72]], i8* %"C'", i8* %[[r35]]
; CHECK-NEXT:   %[[r76:.+]] = select i1 %[[r72]], i8* %cast.ldc, i8* %[[r65]]
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %[[r58]], i8* %[[r59]], i8* %[[r60]], i8* %[[r61]], i8* %[[r37]], i8* %cast.alpha, i8* %[[r73]], i8* %[[r74]], i8* %[[r75]], i8* %[[r76]], i8* %fpcast.constant.fp.1.0, i8* %"B'", i8* %cast.ldb, i64 1, i64 1)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G, align 1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.033, align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %[[fp035]], align 8
; CHECK-NEXT:   %[[fp036:.+]] = bitcast double* %[[fp035]] to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %byref.constant.int.033, i8* %[[fp036]], i8* %cast.beta, i8* %[[r37]], i8* %n_p_unwrap, i8* %"C'", i8* %cast.ldc, i64* %[[tmp]], i64 1)
; CHECK-NEXT:   %[[r77:.+]] = bitcast double* %cache.A_unwrap to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r77]])
; CHECK-NEXT:   call void @free(i8* %[[r37]])
; CHECK-NEXT:   %[[r78:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[r79:.+]] = icmp eq i64 %[[r78]], 0
; CHECK-NEXT:   %[[r80:.+]] = xor i1 %[[r79]], true
; CHECK-NEXT:   br i1 %[[r79]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %[[i72:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[i73:.+]] = add nsw i64 %[[i72]], -1
; CHECK-NEXT:   store i64 %[[i73]], i64* %"iv'ac"
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertexit:                                       ; preds = %exit
; CHECK-NEXT:   br label %mergeinvertloop_exit

; CHECK: mergeinvertloop_exit:                             ; preds = %invertexit
; CHECK-NEXT:   store i64 9, i64* %"iv'ac"
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %mergeinvertloop_exit, %incinvertloop
; CHECK-NEXT:   br label %remat_loop_loop

; CHECK: remat_loop_loop:                                  ; preds = %remat_enter
; CHECK-NEXT:   %remat_m_p = call i8* @malloc(i64 8)
; CHECK-NEXT:   store i8* %remat_m_p, i8** %m_p_cache, align 8, !invariant.group !6
; CHECK-NEXT:   %[[i74:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[i75:.+]] = load i8*, i8** %m_p_cache, align 8, !invariant.group !6
; CHECK-NEXT:   %m_unwrap = bitcast i8* %[[i75]] to i64*
; CHECK-NEXT:   store i64 4, i64* %m_unwrap, align 16
; CHECK-NEXT:   %[[i76:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %iv.next_unwrap = add nuw nsw i64 %[[i76]], 1
; CHECK-NEXT:   %cmp_unwrap = icmp eq i64 %iv.next_unwrap, 10
; CHECK-NEXT:   br i1 %cmp_unwrap, label %invertloop, label %invertloop
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_memcpy_double_mat_64(double* noalias nocapture writeonly %dst, double* noalias nocapture readonly %src, i64 %M, i64 %N, i64 %LDA)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = add nuw nsw i64 %M, %N
; CHECK-NEXT:   %0 = icmp eq i64 %mul, 0
; CHECK-NEXT:   br i1 %0, label %for.end, label %init.idx

; CHECK: init.idx:                                         ; preds = %init.end, %entry
; CHECK-NEXT:   %j = phi i64 [ 0, %entry ], [ %j.next, %init.end ]
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %init.idx
; CHECK-NEXT:   %i = phi i64 [ 0, %init.idx ], [ %i.next, %for.body ]
; CHECK-NEXT:   %1 = mul nuw nsw i64 %j, %M
; CHECK-NEXT:   %2 = add nuw nsw i64 %i, %1
; CHECK-NEXT:   %dst.i = getelementptr inbounds double, double* %dst, i64 %2
; CHECK-NEXT:   %3 = mul nuw nsw i64 %j, %LDA
; CHECK-NEXT:   %4 = add nuw nsw i64 %i, %3
; CHECK-NEXT:   %dst.i1 = getelementptr inbounds double, double* %src, i64 %4
; CHECK-NEXT:   %src.i.l = load double, double* %dst.i1
; CHECK-NEXT:   store double %src.i.l, double* %dst.i
; CHECK-NEXT:   %i.next = add nuw nsw i64 %i, 1
; CHECK-NEXT:   %5 = icmp eq i64 %i.next, %M
; CHECK-NEXT:   br i1 %5, label %init.end, label %for.body

; CHECK: init.end:                                         ; preds = %for.body
; CHECK-NEXT:   %j.next = add nuw nsw i64 %j, 1
; CHECK-NEXT:   %6 = icmp eq i64 %j.next, %N
; CHECK-NEXT:   br i1 %6, label %for.end, label %init.idx

; CHECK: for.end:                                          ; preds = %init.end, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
