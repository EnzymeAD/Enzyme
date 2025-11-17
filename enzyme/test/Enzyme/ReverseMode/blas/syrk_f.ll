;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S -enzyme-detect-readthrow=0 | FileCheck %s

; dsyrk	(	character 	UPLO,
; character 	TRANS,
; integer 	N,
; integer 	K,
; double precision 	ALPHA,
; double precision, dimension(lda,*) 	A,
; integer 	LDA,
; double precision 	BETA,
; double precision, dimension(ldc,*) 	C,
; integer 	LDC
; )

declare void @dsyrk_64_(i8* nocapture readonly, i8* nocapture readonly, i64* nocapture readonly, i64* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i64* nocapture readonly, i8* nocapture readonly, i8* nocapture, i64* nocapture readonly, i64, i64) 

define void @f(i8* %C, i8* %A) {
entry:
  %uplo = alloca i8, align 1
  %trans = alloca i8, align 1
  %n = alloca i64, align 16
  %k = alloca i64, align 16
  %alpha = alloca double, align 16
  %alpha_p = bitcast double* %alpha to i8*
  %lda = alloca i64, align 16
  %beta = alloca double, align 16
  %beta_p = bitcast double* %beta to i8*
  %ldc = alloca i64, align 16
  store i8 85, i8* %uplo, align 1
  store i8 78, i8* %trans, align 1
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  call void @dsyrk_64_(i8* %uplo, i8* %trans, i64* %n, i64* %k, i8* %alpha_p, i8* %A, i64* %lda, i8* %beta_p, i8* %C, i64* %ldc, i64 1, i64 1) 
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA)
  ret void
}

; CHECK: define internal void @diffef(i8* %C, i8* %"C'", i8* %A, i8* %"A'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.char.r = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.char.l = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1 = alloca double, align 8
; CHECK-NEXT:   %byref.for.i = alloca i64, align 8
; CHECK-NEXT:   %byref.FMul = alloca double, align 8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.03 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.1 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.06 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.08 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.110 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.012 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.013 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %0 = alloca i64, align 8
; CHECK-NEXT:   %uplo = alloca i8, align 1
; CHECK-NEXT:   %trans = alloca i8, align 1
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %alpha = alloca double, align 16
; CHECK-NEXT:   %alpha_p = bitcast double* %alpha to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %beta = alloca double, align 16
; CHECK-NEXT:   %beta_p = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   store i8 85, i8* %uplo, align 1
; CHECK-NEXT:   store i8 78, i8* %trans, align 1
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta, align 8
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   call void @dsyrk_64_(i8* %uplo, i8* %trans, i64* %n, i64* %k, i8* %alpha_p, i8* %A, i64* %lda, i8* %beta_p, i8* %C, i64* %ldc, i64 1, i64 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 4
; CHECK-NEXT:   store i8 114, i8* %byref.constant.char.r, align 1
; CHECK-NEXT:   store i8 108, i8* %byref.constant.char.l, align 1
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i1:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[i2:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[i3:.+]] = or i1 %[[i2]], %[[i1]]
; CHECK-NEXT:   %[[i4:.+]] = select i1 %[[i3]], i8* %byref.constant.char.l, i8* %byref.constant.char.r
; CHECK-NEXT:   %ld.row.trans1 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i5:.+]] = icmp eq i8 %ld.row.trans1, 110
; CHECK-NEXT:   %[[i6:.+]] = icmp eq i8 %ld.row.trans1, 78
; CHECK-NEXT:   %[[i7:.+]] = or i1 %[[i6]], %[[i5]]
; CHECK-NEXT:   %[[i8:.+]] = select i1 %[[i7:.+]], i64* %n, i64* %k
; CHECK-NEXT:   %[[i9:.+]] = select i1 %[[i7:.+]], i64* %k, i64* %n
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1 = bitcast double* %byref.constant.fp.1 to i8*
; CHECK-NEXT:   call void @dsymm_64_(i8* %[[i4]], i8* %uplo, i64* %[[i8]], i64* %[[i9]], i8* %alpha_p, i8* %"C'", i64* %ldc, i8* %A, i64* %lda, i8* %fpcast.constant.fp.1, i8* %"A'", i64* %lda, i64 1, i64 1)
; CHECK-NEXT:   %[[i11:.+]] = load i64, i64* %n, align 4
; CHECK-NEXT:   %[[i12:.+]] = icmp eq i64 %[[i11]], 0
; CHECK-NEXT:   br i1 %[[i12]], label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_loop:                                 ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   %[[i13:.+]] = phi i64 [ 0, %invertentry ], [ %[[i14:.+]], %invertentry_loop ]
; CHECK-NEXT:   %[[i14]] = add nuw nsw i64 %[[i13]], 1
; CHECK-NEXT:   store i64 %[[i13]], i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[i17:.+]] = load i64, i64* %ldc, align 4
; CHECK-NEXT:   %[[i22:.+]] = load i64, i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[i26:.+]] = load i64, i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[i23:.+]] = mul i64 %[[i22]], 1
; CHECK-NEXT:   %[[i27:.+]] = mul i64 %[[i26]], %[[i17]]
; CHECK-NEXT:   %[[i28:.+]] = add i64 %[[i23]], %[[i27]]
; CHECK-NEXT:   %[[i15:.+]] = bitcast i8* %"C'" to double*

; CHECK-NEXT:   %[[i29:.+]] = getelementptr double, double* %[[i15]], i64 %[[i28]]
; CHECK-NEXT:   %[[zz:.+]] = bitcast double* %[[i29]] to i8*
; CHECK-NEXT:   %[[i31:.+]] = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %[[i32:.+]] = load double, double* %[[i31]], align 8
; CHECK-NEXT:   %[[mm:.+]] = bitcast i8* %[[zz]] to double*
; CHECK-NEXT:   %[[i30:.+]] = load double, double* %[[mm]], align 8
; CHECK-NEXT:   %[[i33:.+]] = fmul fast double %[[i32]], %[[i30]]
; CHECK-NEXT:   store double %[[i33]], double* %byref.FMul, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   %ld.row.trans2 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i34:.+]] = icmp eq i8 %ld.row.trans2, 110
; CHECK-NEXT:   %[[i35:.+]] = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %[[i36:.+]] = or i1 %[[i35]], %[[i34]]
; CHECK-NEXT:   %[[i37:.+]] = select i1 %[[i36]], i64* %byref.for.i, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.03, align 4
; CHECK-NEXT:   %[[ldrowtrans4:.+]] = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i38:.+]] = icmp eq i8 %[[ldrowtrans5:.+]], 110
; CHECK-NEXT:   %[[i39:.+]] = icmp eq i8 %[[ldrowtrans5:.+]], 78
; CHECK-NEXT:   %[[i40:.+]] = or i1 %[[i39]], %[[i38]]
; CHECK-NEXT:   %[[i41:.+]] = select i1 %[[i40]], i64* %byref.constant.int.03, i64* %byref.for.i
; CHECK-NEXT:   %[[i44:.+]] = load i64, i64* %lda, align 4
; CHECK-NEXT:   %[[i49:.+]] = load i64, i64* %[[i37]], align 4
; CHECK-NEXT:   %[[i53:.+]] = load i64, i64* %[[i41]], align 4
; CHECK-NEXT:   %[[i50:.+]] = mul i64 %[[i49]], 1
; CHECK-NEXT:   %[[i54:.+]] = mul i64 %[[i53]], %[[i44]]
; CHECK-NEXT:   %[[i55:.+]] = add i64 %[[i50]], %[[i54]]
; CHECK-NEXT:   %[[i42:.+]] = bitcast i8* %A to double*
; CHECK-NEXT:   %[[i56:.+]] = getelementptr double, double* %[[i42]], i64 %[[i55]]
; CHECK-NEXT:   %[[zi56:.+]] = bitcast double* %[[i56:.+]] to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1, align 4
; CHECK-NEXT:   %[[ldrowtrans5:.+]] = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i57:.+]] = icmp eq i8 %[[ldrowtrans5]], 110
; CHECK-NEXT:   %[[i58:.+]] = icmp eq i8 %[[ldrowtrans5]], 78
; CHECK-NEXT:   %[[i59:.+]] = or i1 %[[i58]], %[[i57]]
; CHECK-NEXT:   %[[i60:.+]] = select i1 %[[i59]], i64* %lda, i64* %byref.constant.int.1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.06, align 4
; CHECK-NEXT:   %[[ldrowtrans7:.+]] = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i61:.+]] = icmp eq i8 %[[ldrowtrans7]], 110
; CHECK-NEXT:   %[[i62:.+]] = icmp eq i8 %[[ldrowtrans7]], 78
; CHECK-NEXT:   %[[i63:.+]] = or i1 %[[i62]], %[[i61]]
; CHECK-NEXT:   %[[i64:.+]] = select i1 %[[i63]], i64* %byref.for.i, i64* %byref.constant.int.06
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.08, align 4
; CHECK-NEXT:   %[[ldrowtrans9:.+]] = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i65:.+]] = icmp eq i8 %[[ldrowtrans9:.+]], 110
; CHECK-NEXT:   %[[i66:.+]] = icmp eq i8 %[[ldrowtrans9:.+]], 78
; CHECK-NEXT:   %[[i67:.+]] = or i1 %[[i66]], %[[i65]]
; CHECK-NEXT:   %[[i79:.+]] = select i1 %[[i67]], i64* %byref.constant.int.08, i64* %byref.for.i
; CHECK-NEXT:   %[[i71:.+]] = load i64, i64* %lda, align 4
; CHECK-NEXT:   %[[i76:.+]] = load i64, i64* %[[i64]], align 4
; CHECK-NEXT:   %[[i80:.+]] = load i64, i64* %[[i79]], align 4
; CHECK-NEXT:   %[[i77:.+]] = mul i64 %[[i76]], 1
; CHECK-NEXT:   %[[i81:.+]] = mul i64 %[[i80]], %[[i71]]
; CHECK-NEXT:   %[[i82:.+]] = add i64 %[[i77]], %[[i81]]
; CHECK-NEXT:   %[[i69:.+]] = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %[[i83:.+]] = getelementptr double, double* %[[i69]], i64 %[[i82]]
; CHECK-NEXT:   %[[zi83:.+]] = bitcast double* %[[i83]] to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.110, align 4
; CHECK-NEXT:   %[[ldrowtrans15:.+]] = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[i84:.+]] = icmp eq i8 %[[ldrowtrans15:.+]], 110
; CHECK-NEXT:   %[[i85:.+]] = icmp eq i8 %[[ldrowtrans15:.+]], 78
; CHECK-NEXT:   %[[i86:.+]] = or i1 %[[i85]], %[[i84]]
; CHECK-NEXT:   %[[i87:.+]] = select i1 %[[i86]], i64* %lda, i64* %byref.constant.int.110
; CHECK-NEXT:   call void @daxpy_64_(i64* %k, double* %byref.FMul, i8* %[[zi56]], i64* %[[i60]], i8* %[[zi83]], i64* %[[i87]])
; CHECK-NEXT:   %[[i88:.+]] = icmp eq i64 %[[i11]], %[[i14]]
; CHECK-NEXT:   br i1 %[[i88]], label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_end:                                  ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.012, align 4
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.013, align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %uplo, i64* %byref.constant.int.012, i64* %byref.constant.int.013, i8* %fpcast.constant.fp.1.0, i8* %beta_p, i64* %n, i64* %n, i8* %"C'", i64* %ldc, i64* %0, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
