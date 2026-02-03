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
; CHECK: entry:
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
; CHECK-NEXT:   %[[v0:.+]] = alloca i64, align 8
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
; CHECK-NEXT:   %[[v1:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[v2:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[v3:.+]] = or i1 %[[v2]], %[[v1]]
; CHECK-NEXT:   %[[v4:.+]] = select i1 %[[v3]], i8* %byref.constant.char.l, i8* %byref.constant.char.r
; CHECK-NEXT:   %ld.row.trans1 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v5:.+]] = icmp eq i8 %ld.row.trans1, 110
; CHECK-NEXT:   %[[v6:.+]] = icmp eq i8 %ld.row.trans1, 78
; CHECK-NEXT:   %[[v7:.+]] = or i1 %[[v6]], %[[v5]]
; CHECK-NEXT:   %[[v8:.+]] = select i1 %[[v7]], i64* %n, i64* %k
; CHECK-NEXT:   %[[v9:.+]] = select i1 %[[v7]], i64* %k, i64* %n
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1 = bitcast double* %byref.constant.fp.1 to i8*
; CHECK-NEXT:   call void @dsymm_64_(i8* %[[v4]], i8* %uplo, i64* %[[v8]], i64* %[[v9]], i8* %alpha_p, i8* %"C'", i64* %ldc, i8* %A, i64* %lda, i8* %fpcast.constant.fp.1, i8* %"A'", i64* %lda, i64 1, i64 1)
; CHECK-NEXT:   %[[v10:.+]] = load i64, i64* %n, align 4
; CHECK-NEXT:   %[[v11:.+]] = icmp eq i64 %[[v10]], 0
; CHECK-NEXT:   br i1 %[[v11]], label %invertentry_end, label %invertentry_loop
; CHECK: invertentry_loop:                                 ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   %[[v12:.+]] = phi i64 [ 0, %invertentry ], [ %[[v13:[0-9]+]], %invertentry_loop ]
; CHECK-NEXT:   %[[v13]] = add nuw nsw i64 %[[v12]], 1
; CHECK-NEXT:   store i64 %[[v12]], i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[v14:.+]] = load i64, i64* %ldc, align 4
; CHECK-NEXT:   %[[v15:.+]] = load i64, i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[v16:.+]] = load i64, i64* %byref.for.i, align 4
; CHECK-NEXT:   %[[v17:.+]] = mul i64 %[[v15]], 1
; CHECK-NEXT:   %[[v18:.+]] = mul i64 %[[v16]], %[[v14]]
; CHECK-NEXT:   %[[v19:.+]] = add i64 %[[v17]], %[[v18]]
; CHECK-NEXT:   %[[v20:.+]] = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %[[v21:.+]] = getelementptr double, double* %[[v20]], i64 %[[v19]]
; CHECK-NEXT:   %[[v22:.+]] = bitcast double* %[[v21]] to i8*
; CHECK-NEXT:   %[[v23:.+]] = bitcast i8* %alpha_p to double*
; CHECK-NEXT:   %[[v24:.+]] = load double, double* %[[v23]], align 8
; CHECK-NEXT:   %[[v25:.+]] = bitcast i8* %[[v22]] to double*
; CHECK-NEXT:   %[[v26:.+]] = load double, double* %[[v25]], align 8
; CHECK-NEXT:   %[[v27:.+]] = fmul fast double %[[v24]], %[[v26]]
; CHECK-NEXT:   store double %[[v27]], double* %byref.FMul, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 4
; CHECK-NEXT:   %ld.row.trans2 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v28:.+]] = icmp eq i8 %ld.row.trans2, 110
; CHECK-NEXT:   %[[v29:.+]] = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %[[v30:.+]] = or i1 %[[v29]], %[[v28]]
; CHECK-NEXT:   %[[v31:.+]] = select i1 %[[v30]], i64* %byref.for.i, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.03, align 4
; CHECK-NEXT:   %ld.row.trans4 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v32:.+]] = icmp eq i8 %ld.row.trans4, 110
; CHECK-NEXT:   %[[v33:.+]] = icmp eq i8 %ld.row.trans4, 78
; CHECK-NEXT:   %[[v34:.+]] = or i1 %[[v33]], %[[v32]]
; CHECK-NEXT:   %[[v35:.+]] = select i1 %[[v34]], i64* %byref.constant.int.03, i64* %byref.for.i
; CHECK-NEXT:   %[[v36:.+]] = load i64, i64* %lda, align 4
; CHECK-NEXT:   %[[v37:.+]] = load i64, i64* %[[v31]], align 4
; CHECK-NEXT:   %[[v38:.+]] = load i64, i64* %[[v35]], align 4
; CHECK-NEXT:   %[[v39:.+]] = mul i64 %[[v37]], 1
; CHECK-NEXT:   %[[v40:.+]] = mul i64 %[[v38]], %[[v36]]
; CHECK-NEXT:   %[[v41:.+]] = add i64 %[[v39]], %[[v40]]
; CHECK-NEXT:   %[[v42:.+]] = bitcast i8* %A to double*
; CHECK-NEXT:   %[[v43:.+]] = getelementptr double, double* %[[v42]], i64 %[[v41]]
; CHECK-NEXT:   %[[v44:.+]] = bitcast double* %[[v43]] to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1, align 4
; CHECK-NEXT:   %ld.row.trans5 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v45:.+]] = icmp eq i8 %ld.row.trans5, 110
; CHECK-NEXT:   %[[v46:.+]] = icmp eq i8 %ld.row.trans5, 78
; CHECK-NEXT:   %[[v47:.+]] = or i1 %[[v46]], %[[v45]]
; CHECK-NEXT:   %[[v48:.+]] = select i1 %[[v47]], i64* %lda, i64* %byref.constant.int.1
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.06, align 4
; CHECK-NEXT:   %ld.row.trans7 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v49:.+]] = icmp eq i8 %ld.row.trans7, 110
; CHECK-NEXT:   %[[v50:.+]] = icmp eq i8 %ld.row.trans7, 78
; CHECK-NEXT:   %[[v51:.+]] = or i1 %[[v50]], %[[v49]]
; CHECK-NEXT:   %[[v52:.+]] = select i1 %[[v51]], i64* %byref.for.i, i64* %byref.constant.int.06
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.08, align 4
; CHECK-NEXT:   %ld.row.trans9 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v53:.+]] = icmp eq i8 %ld.row.trans9, 110
; CHECK-NEXT:   %[[v54:.+]] = icmp eq i8 %ld.row.trans9, 78
; CHECK-NEXT:   %[[v55:.+]] = or i1 %[[v54]], %[[v53]]
; CHECK-NEXT:   %[[v56:.+]] = select i1 %[[v55]], i64* %byref.constant.int.08, i64* %byref.for.i
; CHECK-NEXT:   %[[v57:.+]] = load i64, i64* %lda, align 4
; CHECK-NEXT:   %[[v58:.+]] = load i64, i64* %[[v52]], align 4
; CHECK-NEXT:   %[[v59:.+]] = load i64, i64* %[[v56]], align 4
; CHECK-NEXT:   %[[v60:.+]] = mul i64 %[[v58]], 1
; CHECK-NEXT:   %[[v61:.+]] = mul i64 %[[v59]], %[[v57]]
; CHECK-NEXT:   %[[v62:.+]] = add i64 %[[v60]], %[[v61]]
; CHECK-NEXT:   %[[v63:.+]] = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %[[v64:.+]] = getelementptr double, double* %[[v63]], i64 %[[v62]]
; CHECK-NEXT:   %[[v65:.+]] = bitcast double* %[[v64]] to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.110, align 4
; CHECK-NEXT:   %ld.row.trans11 = load i8, i8* %trans, align 1
; CHECK-NEXT:   %[[v66:.+]] = icmp eq i8 %ld.row.trans11, 110
; CHECK-NEXT:   %[[v67:.+]] = icmp eq i8 %ld.row.trans11, 78
; CHECK-NEXT:   %[[v68:.+]] = or i1 %[[v67]], %[[v66]]
; CHECK-NEXT:   %[[v69:.+]] = select i1 %[[v68]], i64* %lda, i64* %byref.constant.int.110
; CHECK-NEXT:   call void @daxpy_64_(i64* %k, double* %byref.FMul, i8* %[[v44]], i64* %[[v48]], i8* %[[v65]], i64* %[[v69]])
; CHECK-NEXT:   %[[v70:.+]] = icmp eq i64 %[[v10]], %[[v13]]
; CHECK-NEXT:   br i1 %[[v70]], label %invertentry_end, label %invertentry_loop
; CHECK: invertentry_end:                                  ; preds = %invertentry_loop, %invertentry
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.012, align 4
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.013, align 4
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %uplo, i64* %byref.constant.int.012, i64* %byref.constant.int.013, i8* %fpcast.constant.fp.1.0, i8* %beta_p, i64* %n, i64* %n, i8* %"C'", i64* %ldc, i64* %[[v0]], i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
