;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S | FileCheck %s

declare void @dspmv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly) 

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
  call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %AP, i8* %X, i8* %incx_p, i8* %beta, i8* %Y, i8* %incy_p) 
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
; CHECK-NEXT:   %byref.constant.fp.1.08 = alloca double
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
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %1, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.ap = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @llvm.memcpy.p0f64.p0i8.i64(double* %cache.ap, i8* %AP, i64 %1, i1 false)
; CHECK-NEXT:   %2 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %3 = load i64, i64* %2
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %3, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.
; CHECK-NEXT:   call void @dcopy_64_(i8* %n_p, i8* %Y, i8* %incy_p, double* %cache.y, i64* %byref.)
; CHECK-NEXT:   %4 = insertvalue { double*, double* } undef, double* %cache.ap, 0
; CHECK-NEXT:   %5 = insertvalue { double*, double* } %4, double* %cache.y, 1
; CHECK-NEXT:   %6 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %7 = load i64, i64* %6
; CHECK-NEXT:   %8 = add i64 %7, 1
; CHECK-NEXT:   %square_mat_size_y0 = mul i64 %7, %8
; CHECK-NEXT:   %size_y0 = udiv i64 %square_mat_size_y0, 2
; CHECK-NEXT:   %mallocsize4 = mul nuw nsw i64 %size_y0, 8
; CHECK-NEXT:   %malloccall5 = tail call noalias nonnull i8* @malloc(i64 %mallocsize4)
; CHECK-NEXT:   %mat_y0 = bitcast i8* %malloccall5 to double*
; CHECK-NEXT:   %9 = bitcast double* %mat_y0 to i8*
; CHECK-NEXT:   call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %AP, i8* %X, i8* %incx_p, i8* %beta, i8* %Y, i8* %incy_p)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"AP'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %AP to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %tape.ext.ap = extractvalue { double*, double* } %5, 0
; CHECK-NEXT:   %10 = bitcast double* %tape.ext.ap to i8*
; CHECK-NEXT:   %tape.ext.y = extractvalue { double*, double* } %5, 1
; CHECK-NEXT:   %11 = bitcast double* %tape.ext.y to i8*
; CHECK-NEXT:   %tape.ext.y3 = extractvalue { double*, double* } %5, 1
; CHECK-NEXT:   %12 = bitcast double* %tape.ext.y3 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1
; CHECK-NEXT:   %intcast.constant.int.1 = bitcast i64* %byref.constant.int.1 to i8*
; CHECK-NEXT:   call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %fpcast.constant.fp.1.0, i8* %10, i8* %X, i8* %incx_p, i8* %fpcast.constant.fp.0.0, i8* %9, i8* %intcast.constant.int.1)
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.16
; CHECK-NEXT:   %intcast.constant.int.17 = bitcast i64* %byref.constant.int.16 to i8*
; CHECK-NEXT:   %13 = call fast double @ddot_64_(i8* %n_p, i8* %"Y'", i8* %incy_p, i8* %9, i8* %intcast.constant.int.17)
; CHECK-NEXT:   %14 = bitcast i8* %"alpha'" to double*
; CHECK-NEXT:   %15 = load double, double* %14
; CHECK-NEXT:   %16 = fadd fast double %15, %13
; CHECK-NEXT:   store double %16, double* %14
; CHECK-NEXT:   call void @dspr2_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %X, i8* %incx_p, i8* %"Y'", i8* %incy_p, i8* %"AP'")
; CHECK:   %17 = load i64, i64* %n
; CHECK-NEXT:   %18 = load i64, i64* %incx
; CHECK-NEXT:   %19 = bitcast i8* %"Y'" to i64*
; CHECK-NEXT:   %20 = load i64, i64* %19
; CHECK-NEXT:   %21 = bitcast i8* %alpha to double*
; CHECK-NEXT:   %22 = load double, double* %21
; CHECK-NEXT:   %loaded.trans.i = load i8, i8* %uplo
; CHECK-NEXT:   %23 = icmp eq i8 %loaded.trans.i, 85
; CHECK-NEXT:   %24 = icmp eq i8 %loaded.trans.i, 117
; CHECK-NEXT:   %25 = or i1 %24, %23
; CHECK-NEXT:   %k.i = select i1 %25, i64 0, i64 1
; CHECK-NEXT:   %26 = icmp eq i64 %17, 0
; CHECK-NEXT:   br i1 %26, label %__enzyme_spmv_diagd_64_.exit, label %init.i

; CHECK: init.i:                                           ; preds = %invertentry
; CHECK-NEXT:   %27 = bitcast i8* %X to double*
; CHECK-NEXT:   %28 = bitcast i8* %incx_p to double*
; CHECK-NEXT:   %29 = bitcast i8* %incy_p to double*
; CHECK-NEXT:   br i1 %25, label %uper.i, label %lower.i

; CHECK: uper.i:                                           ; preds = %uper.i, %init.i
; CHECK-NEXT:   %iteration.i = phi i64 [ 0, %init.i ], [ %iter.next.i, %uper.i ]
; CHECK-NEXT:   %k1.i = phi i64 [ 0, %init.i ], [ %k.next.i, %uper.i ]
; CHECK-NEXT:   %iter.next.i = add i64 %iteration.i, 1
; CHECK-NEXT:   %k.next.i = add i64 %k1.i, %iter.next.i
; CHECK-NEXT:   %x.idx.i = mul nuw i64 %iteration.i, %18
; CHECK-NEXT:   %y.idx.i = mul nuw i64 %iteration.i, %20
; CHECK-NEXT:   %x.ptr.i = getelementptr inbounds double, double* %27, i64 %x.idx.i
; CHECK-NEXT:   %y.ptr.i = getelementptr inbounds double, double* %28, i64 %y.idx.i
; CHECK-NEXT:   %x.val.i = load double, double* %x.ptr.i
; CHECK-NEXT:   %y.val.i = load double, double* %y.ptr.i
; CHECK-NEXT:   %xy.i = fmul fast double %x.val.i, %y.val.i
; CHECK-NEXT:   %xy.alpha.i = fmul fast double %xy.i, %22
; CHECK-NEXT:   %k.ptr.i = getelementptr inbounds double, double* %29, i64 %k1.i
; CHECK-NEXT:   %k.val.i = load double, double* %k.ptr.i
; CHECK-NEXT:   %k.val.new.i = fsub fast double %k.val.i, %xy.alpha.i
; CHECK-NEXT:   store double %k.val.new.i, double* %k.ptr.i
; CHECK-NEXT:   %30 = icmp eq i64 %iter.next.i, %17
; CHECK-NEXT:   br i1 %30, label %__enzyme_spmv_diagd_64_.exit, label %uper.i

; CHECK: lower.i:                                          ; preds = %lower.i, %init.i
; CHECK-NEXT:   %iteration2.i = phi i64 [ 0, %init.i ], [ %iter.next4.i, %lower.i ]
; CHECK-NEXT:   %k3.i = phi i64 [ 0, %init.i ], [ %k.next5.i, %lower.i ]
; CHECK-NEXT:   %iter.next4.i = add i64 %iteration2.i, 1
; CHECK-NEXT:   %tmp.val.i = add i64 %17, 1
; CHECK-NEXT:   %tmp.val.other.i = sub i64 %tmp.val.i, %iter.next4.i
; CHECK-NEXT:   %k.next5.i = add i64 %k3.i, %tmp.val.other.i
; CHECK-NEXT:   %x.idx6.i = mul nuw i64 %iteration2.i, %18
; CHECK-NEXT:   %y.idx7.i = mul nuw i64 %iteration2.i, %20
; CHECK-NEXT:   %x.ptr8.i = getelementptr inbounds double, double* %27, i64 %x.idx6.i
; CHECK-NEXT:   %y.ptr9.i = getelementptr inbounds double, double* %28, i64 %y.idx7.i
; CHECK-NEXT:   %x.val10.i = load double, double* %x.ptr8.i
; CHECK-NEXT:   %y.val11.i = load double, double* %y.ptr9.i
; CHECK-NEXT:   %xy12.i = fmul fast double %x.val10.i, %y.val11.i
; CHECK-NEXT:   %xy.alpha13.i = fmul fast double %xy12.i, %22
; CHECK-NEXT:   %k.ptr14.i = getelementptr inbounds double, double* %29, i64 %k3.i
; CHECK-NEXT:   %k.val15.i = load double, double* %k.ptr14.i
; CHECK-NEXT:   %k.val.new16.i = fsub fast double %k.val15.i, %xy.alpha13.i
; CHECK-NEXT:   store double %k.val.new16.i, double* %k.ptr14.i
; CHECK-NEXT:   %31 = icmp eq i64 %iter.next4.i, %17
; CHECK-NEXT:   br i1 %31, label %__enzyme_spmv_diagd_64_.exit, label %lower.i

; CHECK: __enzyme_spmv_diagd_64_.exit:                     ; preds = %invertentry, %uper.i, %lower.i
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.08
; CHECK-NEXT:   %fpcast.constant.fp.1.09 = bitcast double* %byref.constant.fp.1.08 to i8*
; CHECK-NEXT:   call void @dspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %10, i8* %"Y'", i8* %incy_p, i8* %fpcast.constant.fp.1.09, i8* %"X'", i8* %incx_p)
; CHECK-NEXT:   %32 = call fast double @ddot_64_(i8* %n_p, i8* %"Y'", i8* %incy_p, i8* %11, i8* %intcast.int.one)
; CHECK-NEXT:   %33 = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %34 = load double, double* %33
; CHECK-NEXT:   %35 = fadd fast double %34, %32
; CHECK-NEXT:   store double %35, double* %33
; CHECK-NEXT:   call void @dscal_64_(i8* %n_p, i8* %beta, i8* %"Y'", i8* %incy_p)
; CHECK-NEXT:   %36 = bitcast double* %tape.ext.ap to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %36)
; CHECK-NEXT:   %37 = bitcast double* %tape.ext.y3 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %37)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_spmv_diagd_64_(i8* %blasuplo, i8* %blasn, i8* noalias nocapture readonly %blasalpha, i8* noalias nocapture readonly %blasx, i8* %blasdy, i8* noalias nocapture readonly %blasincy, i8* %blasdAP, i8* noalias nocapture
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = bitcast i8* %blasn to i64*
; CHECK-NEXT:   %2 = load i64, i64* %1
; CHECK-NEXT:   %3 = bitcast i8* %blasdy to i64*
; CHECK-NEXT:   %4 = load i64, i64* %3
; CHECK-NEXT:   %5 = bitcast i8* %blasincy to i64*
; CHECK-NEXT:   %6 = load i64, i64* %5
; CHECK-NEXT:   %7 = bitcast i8* %blasalpha to double*
; CHECK-NEXT:   %8 = load double, double* %7
; CHECK-NEXT:   %loaded.trans = load i8, i8* %blasuplo
; CHECK-NEXT:   %9 = icmp eq i8 %loaded.trans, 85
; CHECK-NEXT:   %10 = icmp eq i8 %loaded.trans, 117
; CHECK-NEXT:   %11 = or i1 %10, %9
; CHECK-NEXT:   %k = select i1 %11, i64 0, i64 1
; CHECK-NEXT:   %12 = icmp eq i64 %2, 0
; CHECK-NEXT:   br i1 %12, label %for.end, label %init

; CHECK: init:                                             ; preds = %entry
; CHECK-NEXT:   %13 = bitcast i8* %blasx to double*
; CHECK-NEXT:   %14 = bitcast i8* %blasdy to double*
; CHECK-NEXT:   %15 = bitcast i8* %blasdAP to double*
; CHECK-NEXT:   br i1 %11, label %uper, label %lower

; CHECK: uper:                                             ; preds = %uper, %init
; CHECK-NEXT:   %iteration = phi i64 [ 0, %init ], [ %iter.next, %uper ]
; CHECK-NEXT:   %k1 = phi i64 [ 0, %init ], [ %k.next, %uper ]
; CHECK-NEXT:   %iter.next = add i64 %iteration, 1
; CHECK-NEXT:   %k.next = add i64 %k1, %iter.next
; CHECK-NEXT:   %x.idx = mul nuw i64 %iteration, %4
; CHECK-NEXT:   %y.idx = mul nuw i64 %iteration, %6
; CHECK-NEXT:   %x.ptr = getelementptr inbounds double, double* %13, i64 %x.idx
; CHECK-NEXT:   %y.ptr = getelementptr inbounds double, double* %14, i64 %y.idx
; CHECK-NEXT:   %x.val = load double, double* %x.ptr
; CHECK-NEXT:   %y.val = load double, double* %y.ptr
; CHECK-NEXT:   %xy = fmul fast double %x.val, %y.val
; CHECK-NEXT:   %xy.alpha = fmul fast double %xy, %8
; CHECK-NEXT:   %k.ptr = getelementptr inbounds double, double* %15, i64 %k1
; CHECK-NEXT:   %k.val = load double, double* %k.ptr
; CHECK-NEXT:   %k.val.new = fsub fast double %k.val, %xy.alpha
; CHECK-NEXT:   store double %k.val.new, double* %k.ptr
; CHECK-NEXT:   %16 = icmp eq i64 %iter.next, %2
; CHECK-NEXT:   br i1 %16, label %for.end, label %uper

; CHECK: lower:                                            ; preds = %lower, %init
; CHECK-NEXT:   %iteration2 = phi i64 [ 0, %init ], [ %iter.next4, %lower ]
; CHECK-NEXT:   %k3 = phi i64 [ 0, %init ], [ %k.next5, %lower ]
; CHECK-NEXT:   %iter.next4 = add i64 %iteration2, 1
; CHECK-NEXT:   %tmp.val = add i64 %2, 1
; CHECK-NEXT:   %tmp.val.other = sub i64 %tmp.val, %iter.next4
; CHECK-NEXT:   %k.next5 = add i64 %k3, %tmp.val.other
; CHECK-NEXT:   %x.idx6 = mul nuw i64 %iteration2, %4
; CHECK-NEXT:   %y.idx7 = mul nuw i64 %iteration2, %6
; CHECK-NEXT:   %x.ptr8 = getelementptr inbounds double, double* %13, i64 %x.idx6
; CHECK-NEXT:   %y.ptr9 = getelementptr inbounds double, double* %14, i64 %y.idx7
; CHECK-NEXT:   %x.val10 = load double, double* %x.ptr8
; CHECK-NEXT:   %y.val11 = load double, double* %y.ptr9
; CHECK-NEXT:   %xy12 = fmul fast double %x.val10, %y.val11
; CHECK-NEXT:   %xy.alpha13 = fmul fast double %xy12, %8
; CHECK-NEXT:   %k.ptr14 = getelementptr inbounds double, double* %15, i64 %k3
; CHECK-NEXT:   %k.val15 = load double, double* %k.ptr14
; CHECK-NEXT:   %k.val.new16 = fsub fast double %k.val15, %xy.alpha13
; CHECK-NEXT:   store double %k.val.new16, double* %k.ptr14
; CHECK-NEXT:   %17 = icmp eq i64 %iter.next4, %2
; CHECK-NEXT:   br i1 %17, label %for.end, label %lower

; CHECK: for.end:                                          ; preds = %lower, %uper, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
