;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S -enzyme-detect-readthrow=0 | FileCheck %s

declare void @dspmv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i64) 

; 	character  	UPLO,
;		integer  	N,
;		real  	ALPHA,
;		real, dimension(*)  	AP,
;		real, dimension(*)  	X,
;		integer  	INCX,
;		real  	BETA,
;		real, dimension(*)  	Y,
;		integer  	INCY
;
define void @f(i8* noalias %AP, i8* noalias %X, i8* noalias %Y, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %uplo = alloca i8, align 1
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %incx = alloca i64, align 16
  %incx_p = bitcast i64* %incx to i8*
  %incy = alloca i64, align 16
  %incy_p = bitcast i64* %incy to i8*
  ; 85 = U
  store i8 85, i8* %uplo, align 1
  store i64 4, i64* %n, align 16
  store i64 2, i64* %incx, align 16
  store i64 1, i64* %incy, align 16
  call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %AP, i8* %X, i8* %incx_p, i8* %beta, i8* %Y, i8* %incy_p, i64 1) 
  %ptr = bitcast i8* %AP to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %AP, i8* %dAP, i8* %X, i8* %dX, i8* %Y, i8* %dY, i8* %alpha, i8* %dalpha, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %AP, i8* %dAP, metadata !"enzyme_dup", i8* %X, i8* %dX, metadata !"enzyme_dup", i8* %Y, i8* %dY, metadata !"enzyme_dup", i8* %alpha, i8* %dalpha, metadata !"enzyme_dup", i8* %beta, i8* %dbeta)
  ret void
}


; CHECK: define internal void @diffef(i8* noalias %AP, i8* %"AP'", i8* noalias %X, i8* %"X'", i8* noalias %Y, i8* %"Y'", i8* noalias %alpha, i8* %"alpha'", i8* noalias %beta, i8* %"beta'"
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref. = alloca i64
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double
; CHECK-NEXT:   %byref.constant.int.1 = alloca i64
; CHECK-NEXT:   %byref.constant.int.16 = alloca i64
; CHECK-NEXT:   %[[fp107:.+]] = alloca double
; CHECK-NEXT:   %uplo = alloca i8, align 1
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %incx = alloca i64, align 16
; CHECK-NEXT:   %incx_p = bitcast i64* %incx to i8*
; CHECK-NEXT:   %incy = alloca i64, align 16
; CHECK-NEXT:   %incy_p = bitcast i64* %incy to i8*
; CHECK-NEXT:   store i8 85, i8* %uplo, align 1
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 2, i64* %incx, align 16
; CHECK-NEXT:   store i64 1, i64* %incy, align 16
; CHECK-NEXT:   %0 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %1 = load i64, i64* %0
; CHECK-NEXT:   %[[npos_cmp:.+]] = icmp sgt i64 %1, 0
; CHECK-NEXT:   %[[npos:.+]] = select i1 %[[npos_cmp]], i64 %1, i64 0
; CHECK-NEXT:   %[[nplus1:.+]] = add i64 %[[npos]], 1
; CHECK-NEXT:   %[[nprod:.+]] = mul i64 %[[npos]], %[[nplus1]]
; CHECK-NEXT:   %[[apelems:.+]] = lshr i64 %[[nprod]], 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[apelems]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.ap = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %[[apbytes:.+]] = mul i64 %[[apelems]], 8
; CHECK-NEXT:   call void @llvm.memcpy.p0f64.p0i8.i64(double* %cache.ap, i8* %AP, i64 %[[apbytes]], i1 false)
; CHECK-NEXT:   %[[np2:.+]] = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %[[n2:.+]] = load i64, i64* %[[np2]]
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %[[n2]], 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.
; CHECK-NEXT:   call void @dcopy_64_(i8* %n_p, i8* %Y, i8* %incy_p, double* %cache.y, i64* %byref.)
; CHECK-NEXT:   call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %AP, i8* %X, i8* %incx_p, i8* %beta, i8* %Y, i8* %incy_p, i64 1)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"AP'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %AP to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope ![[scope:[0-9]+]], !noalias ![[noalias:[0-9]+]]
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope ![[noalias]], !noalias ![[scope]]
; CHECK-NEXT:   %[[i10:.+]] = bitcast double* %cache.ap to i8*
; CHECK-NEXT:   %[[i11:.+]] = bitcast double* %cache.y to i8*
; CHECK-NEXT:   %[[i12:.+]] = bitcast double* %cache.y to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one

; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %mallocsize4 = mul nuw nsw i64 %[[i7]], 8
; CHECK-NEXT:   %malloccall5 = tail call noalias nonnull i8* @malloc(i64 %mallocsize4)
; CHECK-NEXT:   %[[mat_y0:.+]] = bitcast i8* %malloccall5 to double*
; CHECK-NEXT:   %[[i9:.+]] = bitcast double* %[[mat_y0]] to i8*

; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1
; CHECK-NEXT:   call void bitcast (void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64)* @dspmv_64_ to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64*, i64)*)(i8* %uplo, i8* %n_p, i8* %fpcast.constant.fp.1.0, i8* %[[i10]], i8* %X, i8* %incx_p, i8* %fpcast.constant.fp.0.0, i8* %[[i9]], i64* %byref.constant.int.1, i64 1)
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.16
; CHECK-NEXT:   %[[i13:.+]] = call fast double @ddot_64_(i8* %n_p, i8* %"Y'", i8* %incy_p, i8* %[[i9]], i64* %byref.constant.int.16)
; CHECK-NEXT:   %[[i14:.+]] = bitcast i8* %"alpha'" to double*
; CHECK-NEXT:   %[[i15:.+]] = load double, double* %[[i14]]
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %[[i15]], %[[i13]]
; CHECK-NEXT:   store double %[[i16]], double* %[[i14]]
; CHECK-NEXT:   %[[forfree:.+]] = bitcast double* %[[mat_y0]] to i8* 
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[forfree]])
; CHECK-NEXT:   call void @dspr2_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %X, i8* %incx_p, i8* %"Y'", i8* %incy_p, i8* %"AP'", i64 1)
; CHECK:   %[[i17:.+]] = load i64, i64* %n
; CHECK-NEXT:   %[[i18:.+]] = load i64, i64* %incx
; CHECK-NEXT:   %[[i20:.+]] = load i64, i64* %incy
; CHECK-NEXT:   %[[i21:.+]] = bitcast i8* %alpha to double*
; CHECK-NEXT:   %[[i22:.+]] = load double, double* %[[i21]]
; CHECK-NEXT:   %loaded.trans.i = load i8, i8* %uplo
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans.i, 76
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans.i, 108
; CHECK-NEXT:   %[[i25:.+]] = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i26:.+]] = icmp sle i64 %[[i17]], 0
; CHECK-NEXT:   br i1 %[[i26]], label %__enzyme_spmv_diagd_64_.exit, label %init.i

; CHECK: init.i:                                           ; preds = %invertentry
; CHECK-NEXT:   %[[i27:.+]] = bitcast i8* %X to double*
; CHECK-NEXT:   %[[i28:.+]] = bitcast i8* %"Y'" to double*
; CHECK-NEXT:   %[[i29:.+]] = bitcast i8* %"AP'" to double*
; CHECK-NEXT:   %one.minus.n.i = sub i64 1, %[[i17]]
; CHECK-NEXT:   %[[xmul:.+]] = mul i64 %one.minus.n.i, %[[i18]]
; CHECK-NEXT:   %[[xcmp:.+]] = icmp slt i64 %[[i18]], 0
; CHECK-NEXT:   %x.start.i = select i1 %[[xcmp]], i64 %[[xmul]], i64 0
; CHECK-NEXT:   %[[ymul:.+]] = mul i64 %one.minus.n.i, %[[i20]]
; CHECK-NEXT:   %[[ycmp:.+]] = icmp slt i64 %[[i20]], 0
; CHECK-NEXT:   %y.start.i = select i1 %[[ycmp]], i64 %[[ymul]], i64 0
; CHECK-NEXT:   br i1 %[[i25]], label %lower.i, label %uper.i 

; CHECK: uper.i:                                           ; preds = %uper.i, %init.i
; CHECK-NEXT:   %iteration.i = phi i64 [ 0, %init.i ], [ %iter.next.i, %uper.i ]
; CHECK-NEXT:   %k.i = phi i64 [ 0, %init.i ], [ %k.next.i, %uper.i ]
; CHECK-NEXT:   %x.idx.i = phi i64 [ %x.start.i, %init.i ], [ %x.next.i, %uper.i ]
; CHECK-NEXT:   %y.idx.i = phi i64 [ %y.start.i, %init.i ], [ %y.next.i, %uper.i ]
; CHECK-NEXT:   %iter.next.i = add i64 %iteration.i, 1
; CHECK-NEXT:   %k.step.i = add i64 %iter.next.i, 1
; CHECK-NEXT:   %k.next.i = add i64 %k.i, %k.step.i
; CHECK-NEXT:   %x.next.i = add i64 %x.idx.i, %[[i18]]
; CHECK-NEXT:   %y.next.i = add i64 %y.idx.i, %[[i20]]
; CHECK-NEXT:   %x.ptr.i = getelementptr inbounds double, double* %[[i27]], i64 %x.idx.i
; CHECK-NEXT:   %y.ptr.i = getelementptr inbounds double, double* %[[i28]], i64 %y.idx.i
; CHECK-NEXT:   %x.val.i = load double, double* %x.ptr.i
; CHECK-NEXT:   %y.val.i = load double, double* %y.ptr.i
; CHECK-NEXT:   %xy.i = fmul fast double %x.val.i, %y.val.i
; CHECK-NEXT:   %xy.alpha.i = fmul fast double %xy.i, %[[i22]]
; CHECK-NEXT:   %k.ptr.i = getelementptr inbounds double, double* %[[i29]], i64 %k.i
; CHECK-NEXT:   %k.val.i = load double, double* %k.ptr.i
; CHECK-NEXT:   %k.val.new.i = fsub fast double %k.val.i, %xy.alpha.i
; CHECK-NEXT:   store double %k.val.new.i, double* %k.ptr.i
; CHECK-NEXT:   %[[i30:.+]] = icmp eq i64 %iter.next.i, %[[i17]]
; CHECK-NEXT:   br i1 %[[i30]], label %__enzyme_spmv_diagd_64_.exit, label %uper.i

; CHECK: lower.i:                                          ; preds = %lower.i, %init.i
; CHECK-NEXT:   %[[iteration2:.+]] = phi i64 [ 0, %init.i ], [ %[[next4:.+]], %lower.i ]
; CHECK-NEXT:   %[[k3:.+]] = phi i64 [ 0, %init.i ], [ %[[next5:.+]], %lower.i ]
; CHECK-NEXT:   %[[idx6:.+]] = phi i64 [ %x.start.i, %init.i ], [ %[[xnext6:.+]], %lower.i ]
; CHECK-NEXT:   %[[idx7:.+]] = phi i64 [ %y.start.i, %init.i ], [ %[[ynext7:.+]], %lower.i ]
; CHECK-NEXT:   %[[next4]] = add i64 %[[iteration2]], 1
; CHECK-NEXT:   %[[kstep:.+]] = sub i64 %[[i17]], %[[iteration2]]
; CHECK-NEXT:   %[[next5]] = add i64 %[[k3]], %[[kstep]]
; CHECK-NEXT:   %[[xnext6]] = add i64 %[[idx6]], %[[i18]]
; CHECK-NEXT:   %[[ynext7]] = add i64 %[[idx7]], %[[i20]]
; CHECK-NEXT:   %[[ptr8:.+]] = getelementptr inbounds double, double* %[[i27]], i64 %[[idx6]]
; CHECK-NEXT:   %[[ptr9:.+]] = getelementptr inbounds double, double* %[[i28]], i64 %[[idx7]]
; CHECK-NEXT:   %[[val10:.+]] = load double, double* %[[ptr8]]
; CHECK-NEXT:   %[[val11:.+]] = load double, double* %[[ptr9]]
; CHECK-NEXT:   %[[xy12:.+]] = fmul fast double %[[val10]], %[[val11]]
; CHECK-NEXT:   %[[alpha13:.+]] = fmul fast double %[[xy12]], %[[i22]]
; CHECK-NEXT:   %[[ptr14:.+]] = getelementptr inbounds double, double* %[[i29]], i64 %[[k3]]
; CHECK-NEXT:   %[[val15:.+]] = load double, double* %[[ptr14]]
; CHECK-NEXT:   %[[new16:.+]] = fsub fast double %[[val15]], %[[alpha13]]
; CHECK-NEXT:   store double %[[new16]], double* %[[ptr14]]
; CHECK-NEXT:   %[[i31:.+]] = icmp eq i64 %[[next4]], %[[i17]]
; CHECK-NEXT:   br i1 %[[i31]], label %__enzyme_spmv_diagd_64_.exit, label %lower.i

; CHECK: __enzyme_spmv_diagd_64_.exit:                     ; preds = %invertentry, %uper.i, %lower.i
; CHECK-NEXT:   store double 1.000000e+00, double* %[[fp107]]
; CHECK-NEXT:   %[[fp108:.+]] = bitcast double* %[[fp107]] to i8*
; CHECK-NEXT:   call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %[[i10]], i8* %"Y'", i8* %incy_p, i8* %[[fp108]], i8* %"X'", i8* %incx_p, i64 1)
; CHECK-NEXT:   %[[i32:.+]] = call fast double @ddot_64_(i8* %n_p, i8* %"Y'", i8* %incy_p, i8* %[[i11]], i64* %byref.int.one)
; CHECK-NEXT:   %[[i33:.+]] = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %[[i34:.+]] = load double, double* %[[i33]]
; CHECK-NEXT:   %[[i35:.+]] = fadd fast double %[[i34]], %[[i32]]
; CHECK-NEXT:   store double %[[i35]], double* %[[i33]]
; CHECK-NEXT:   call void @dscal_64_(i8* %n_p, i8* %beta, i8* %"Y'", i8* %incy_p)
; CHECK-NEXT:   %[[i36:.+]] = bitcast double* %cache.ap to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i36]])
; CHECK-NEXT:   %[[i37:.+]] = bitcast double* %cache.y to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i37]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_spmv_diagd_64_(i8* %blasuplo, i8* %blasn, i8* noalias nocapture readonly %blasalpha, i8* noalias nocapture readonly %blasx, i8* %blasincx, i8* noalias nocapture readonly %blasdy, i8* %blasincy, i8* noalias nocapture %blasdAP)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[f0:.+]] = bitcast i8* %blasn to i64*
; CHECK-NEXT:   %[[fn:.+]] = load i64, i64* %[[f0]]
; CHECK-NEXT:   %[[f2:.+]] = bitcast i8* %blasincx to i64*
; CHECK-NEXT:   %[[fincx:.+]] = load i64, i64* %[[f2]]
; CHECK-NEXT:   %[[f4:.+]] = bitcast i8* %blasincy to i64*
; CHECK-NEXT:   %[[fincy:.+]] = load i64, i64* %[[f4]]
; CHECK-NEXT:   %[[f6:.+]] = bitcast i8* %blasalpha to double*
; CHECK-NEXT:   %[[falpha:.+]] = load double, double* %[[f6]]
; CHECK-NEXT:   %loaded.trans = load i8, i8* %blasuplo
; CHECK-DAG:   %[[i9:.+]] = icmp eq i8 %loaded.trans, 76
; CHECK-DAG:   %[[i10:.+]] = icmp eq i8 %loaded.trans, 108
; CHECK-NEXT:   %[[fisl:.+]] = or i1 %[[i10]], %[[i9]]
; CHECK-NEXT:   %[[fnle0:.+]] = icmp sle i64 %[[fn]], 0
; CHECK-NEXT:   br i1 %[[fnle0]], label %for.end, label %init

; CHECK: init:                                             ; preds = %entry
; CHECK-NEXT:   %[[fx:.+]] = bitcast i8* %blasx to double*
; CHECK-NEXT:   %[[fdy:.+]] = bitcast i8* %blasdy to double*
; CHECK-NEXT:   %[[fdap:.+]] = bitcast i8* %blasdAP to double*
; CHECK-NEXT:   %one.minus.n = sub i64 1, %[[fn]]
; CHECK-NEXT:   %[[fxm:.+]] = mul i64 %one.minus.n, %[[fincx]]
; CHECK-NEXT:   %[[fxc:.+]] = icmp slt i64 %[[fincx]], 0
; CHECK-NEXT:   %x.start = select i1 %[[fxc]], i64 %[[fxm]], i64 0
; CHECK-NEXT:   %[[fym:.+]] = mul i64 %one.minus.n, %[[fincy]]
; CHECK-NEXT:   %[[fyc:.+]] = icmp slt i64 %[[fincy]], 0
; CHECK-NEXT:   %y.start = select i1 %[[fyc]], i64 %[[fym]], i64 0
; CHECK-NEXT:   br i1 %[[fisl]], label %lower, label %uper

; CHECK: uper:                                             ; preds = %uper, %init
; CHECK-NEXT:   %iteration = phi i64 [ 0, %init ], [ %iter.next, %uper ]
; CHECK-NEXT:   %[[k1:.+]] = phi i64 [ 0, %init ], [ %k.next, %uper ]
; CHECK-NEXT:   %x.idx = phi i64 [ %x.start, %init ], [ %x.next, %uper ]
; CHECK-NEXT:   %y.idx = phi i64 [ %y.start, %init ], [ %y.next, %uper ]
; CHECK-NEXT:   %iter.next = add i64 %iteration, 1
; CHECK-NEXT:   %k.step = add i64 %iter.next, 1
; CHECK-NEXT:   %k.next = add i64 %[[k1]], %k.step
; CHECK-NEXT:   %x.next = add i64 %x.idx, %[[fincx]]
; CHECK-NEXT:   %y.next = add i64 %y.idx, %[[fincy]]
; CHECK-NEXT:   %x.ptr = getelementptr inbounds double, double* %[[fx]], i64 %x.idx
; CHECK-NEXT:   %y.ptr = getelementptr inbounds double, double* %[[fdy]], i64 %y.idx
; CHECK-NEXT:   %x.val = load double, double* %x.ptr
; CHECK-NEXT:   %y.val = load double, double* %y.ptr
; CHECK-NEXT:   %xy = fmul fast double %x.val, %y.val
; CHECK-NEXT:   %xy.alpha = fmul fast double %xy, %[[falpha]]
; CHECK-NEXT:   %k.ptr = getelementptr inbounds double, double* %[[fdap]], i64 %[[k1]]
; CHECK-NEXT:   %k.val = load double, double* %k.ptr
; CHECK-NEXT:   %k.val.new = fsub fast double %k.val, %xy.alpha
; CHECK-NEXT:   store double %k.val.new, double* %k.ptr
; CHECK-NEXT:   %[[f16:.+]] = icmp eq i64 %iter.next, %[[fn]]
; CHECK-NEXT:   br i1 %[[f16]], label %for.end, label %uper

; CHECK: lower:                                            ; preds = %lower, %init
; CHECK-NEXT:   %[[iteration2:.+]] = phi i64 [ 0, %init ], [ %[[next4:.+]], %lower ]
; CHECK-NEXT:   %[[k3:.+]] = phi i64 [ 0, %init ], [ %[[next5:.+]], %lower ]
; CHECK-NEXT:   %[[idx6:.+]] = phi i64 [ %x.start, %init ], [ %[[xnext6:.+]], %lower ]
; CHECK-NEXT:   %[[idx7:.+]] = phi i64 [ %y.start, %init ], [ %[[ynext7:.+]], %lower ]
; CHECK-NEXT:   %[[next4]] = add i64 %[[iteration2]], 1
; CHECK-NEXT:   %[[kstep:.+]] = sub i64 %[[fn]], %[[iteration2]]
; CHECK-NEXT:   %[[next5]] = add i64 %[[k3]], %[[kstep]]
; CHECK-NEXT:   %[[xnext6]] = add i64 %[[idx6]], %[[fincx]]
; CHECK-NEXT:   %[[ynext7]] = add i64 %[[idx7]], %[[fincy]]
; CHECK-NEXT:   %[[ptr8:.+]] = getelementptr inbounds double, double* %[[fx]], i64 %[[idx6]]
; CHECK-NEXT:   %[[ptr9:.+]] = getelementptr inbounds double, double* %[[fdy]], i64 %[[idx7]]
; CHECK-NEXT:   %[[val10:.+]] = load double, double* %[[ptr8]]
; CHECK-NEXT:   %[[val11:.+]] = load double, double* %[[ptr9]]
; CHECK-NEXT:   %[[xy12:.+]] = fmul fast double %[[val10]], %[[val11]]
; CHECK-NEXT:   %[[alpha13:.+]] = fmul fast double %[[xy12]], %[[falpha]]
; CHECK-NEXT:   %[[ptr14:.+]] = getelementptr inbounds double, double* %[[fdap]], i64 %[[k3]]
; CHECK-NEXT:   %[[val15:.+]] = load double, double* %[[ptr14]]
; CHECK-NEXT:   %[[new16:.+]] = fsub fast double %[[val15]], %[[alpha13]]
; CHECK-NEXT:   store double %[[new16]], double* %[[ptr14]]
; CHECK-NEXT:   %[[f17:.+]] = icmp eq i64 %[[next4]], %[[fn]]
; CHECK-NEXT:   br i1 %[[f17]], label %for.end, label %lower

; CHECK: for.end:                                          ; preds = %lower, %uper, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
