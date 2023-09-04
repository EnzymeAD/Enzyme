;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=0 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=0 -S | FileCheck %s


; Function Attrs: nounwind uwtable
define double @coupled_springs(double* noundef %K, double* nocapture readnone %m, double* noundef %x0, double* noundef %v0, double %T, i32 noundef %N) #0 {
entry:
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #10
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #10
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #10
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #10
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #10
  %0 = load double, double* %x0, align 8
  ret double %0
}


declare void @cblas_dgemv(i32 noundef, i32 noundef, i32 noundef, i32 noundef, double noundef, double* noundef, i32 noundef, double* noundef, i32 noundef, double noundef, double* noundef, i32 noundef) local_unnamed_addr #1

declare void @__enzyme_autodiff(i8* noundef, ...) local_unnamed_addr #1

define void @active(double* noundef %K, double* noundef %dK, double* nocapture readnone %m, double* noundef %x0, double* noundef %dx0, double* noundef %v0, double* noundef %dv0, double %T, i32 noundef %N) #0 {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* noundef nonnull bitcast (double (double*, double*, double*, double*, double, i32)* @coupled_springs to i8*), metadata !"enzyme_dup", double* noundef %K, double* noundef %dK, 
  metadata !"enzyme_const", double* noundef %m, metadata !"enzyme_dup", double* noundef %x0, double* noundef %dx0, metadata !"enzyme_dup", double* noundef %v0, double* noundef %dv0, metadata !"enzyme_const", double noundef 1.000000e+00, i32 noundef 15) #4
  ret void
}

; CHECK: define internal void @diffecoupled_springs(double* noundef %K, double* %"K'", double* nocapture readnone %m, double* noundef %x0, double* %"x0'", double* noundef %v0, double* %"v0'", double %T, i32 noundef %N, double 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %0 = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize19 = mul nuw nsw i32 %0, 8
; CHECK-NEXT:   %malloccall20 = tail call noalias nonnull i8* @malloc(i32 %mallocsize19)
; CHECK-NEXT:   %cache.A21 = bitcast i8* %malloccall20 to double*
; CHECK:   %mul.i = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %1 = icmp eq i32 %mul.i, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_mat_32.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %init.end.i, %entry
; CHECK-NEXT:   %j.i = phi i32 [ 0, %entry ], [ %j.next.i, %init.end.i ]
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %i.i = phi i32 [ 0, %init.idx.i ], [ %i.next.i, %for.body.i ]
; CHECK-NEXT:   %2 = mul nuw nsw i32 %j.i, %N
; CHECK-NEXT:   %3 = add nuw nsw i32 %i.i, %2
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %cache.A21, i32 %3
; CHECK-NEXT:   %4 = mul nuw nsw i32 %j.i, %N
; CHECK-NEXT:   %5 = add nuw nsw i32 %i.i, %4
; CHECK-NEXT:   %dst.i1.i = getelementptr inbounds double, double* %K, i32 %5
; CHECK-NEXT:   %src.i.l.i = load double, double* %dst.i1.i, align 8
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i, align 8
; CHECK-NEXT:   %i.next.i = add nuw nsw i32 %i.i, 1
; CHECK-NEXT:   %6 = icmp eq i32 %i.next.i, %N
; CHECK-NEXT:   br i1 %6, label %init.end.i, label %for.body.i

; CHECK: init.end.i:                                       ; preds = %for.body.i
; CHECK-NEXT:   %j.next.i = add nuw nsw i32 %j.i, 1
; CHECK-NEXT:   %7 = icmp eq i32 %j.next.i, %N
; CHECK-NEXT:   br i1 %7, label %__enzyme_memcpy_double_mat_32.exit, label %init.idx.i

; CHECK: __enzyme_memcpy_double_mat_32.exit:               ; preds = %entry, %init.end.i
; CHECK-NEXT:   %8 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   %mallocsize22 = mul nuw nsw i32 %8, 8
; CHECK-NEXT:   %malloccall23 = tail call noalias nonnull i8* @malloc(i32 %mallocsize22)
; CHECK-NEXT:   %cache.x24 = bitcast i8* %malloccall23 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %8, double* %x0, i32 1, double* %cache.x24, i32 1)
; CHECK-NEXT:   %9 = insertvalue { double*, double* } undef, double* %cache.A21, 0
; CHECK-NEXT:   %10 = insertvalue { double*, double* } %9, double* %cache.x24, 1
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %11 = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize11 = mul nuw nsw i32 %11, 8
; CHECK-NEXT:   %malloccall12 = tail call noalias nonnull i8* @malloc(i32 %mallocsize11)
; CHECK-NEXT:   %cache.A13 = bitcast i8* %malloccall12 to double*
; CHECK:   %mul.i27 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %12 = icmp eq i32 %mul.i27, 0
; CHECK-NEXT:   br i1 %12, label %__enzyme_memcpy_double_mat_32.exit38, label %init.idx.i29

; CHECK: init.idx.i29:                                     ; preds = %init.end.i37, %__enzyme_memcpy_double_mat_32.exit
; CHECK-NEXT:   %j.i28 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit ], [ %j.next.i36, %init.end.i37 ]
; CHECK-NEXT:   br label %for.body.i35

; CHECK: for.body.i35:                                     ; preds = %for.body.i35, %init.idx.i29
; CHECK-NEXT:   %i.i30 = phi i32 [ 0, %init.idx.i29 ], [ %i.next.i34, %for.body.i35 ]
; CHECK-NEXT:   %13 = mul nuw nsw i32 %j.i28, %N
; CHECK-NEXT:   %14 = add nuw nsw i32 %i.i30, %13
; CHECK-NEXT:   %dst.i.i31 = getelementptr inbounds double, double* %cache.A13, i32 %14
; CHECK-NEXT:   %15 = mul nuw nsw i32 %j.i28, %N
; CHECK-NEXT:   %16 = add nuw nsw i32 %i.i30, %15
; CHECK-NEXT:   %dst.i1.i32 = getelementptr inbounds double, double* %K, i32 %16
; CHECK-NEXT:   %src.i.l.i33 = load double, double* %dst.i1.i32, align 8
; CHECK-NEXT:   store double %src.i.l.i33, double* %dst.i.i31, align 8
; CHECK-NEXT:   %i.next.i34 = add nuw nsw i32 %i.i30, 1
; CHECK-NEXT:   %17 = icmp eq i32 %i.next.i34, %N
; CHECK-NEXT:   br i1 %17, label %init.end.i37, label %for.body.i35

; CHECK: init.end.i37:                                     ; preds = %for.body.i35
; CHECK-NEXT:   %j.next.i36 = add nuw nsw i32 %j.i28, 1
; CHECK-NEXT:   %18 = icmp eq i32 %j.next.i36, %N
; CHECK-NEXT:   br i1 %18, label %__enzyme_memcpy_double_mat_32.exit38, label %init.idx.i29

; CHECK: __enzyme_memcpy_double_mat_32.exit38:             ; preds = %__enzyme_memcpy_double_mat_32.exit, %init.end.i37
; CHECK-NEXT:   %19 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   %mallocsize14 = mul nuw nsw i32 %19, 8
; CHECK-NEXT:   %malloccall15 = tail call noalias nonnull i8* @malloc(i32 %mallocsize14)
; CHECK-NEXT:   %cache.x16 = bitcast i8* %malloccall15 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %19, double* %x0, i32 1, double* %cache.x16, i32 1)
; CHECK-NEXT:   %20 = insertvalue { double*, double* } undef, double* %cache.A13, 0
; CHECK-NEXT:   %21 = insertvalue { double*, double* } %20, double* %cache.x16, 1
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %22 = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize3 = mul nuw nsw i32 %22, 8
; CHECK-NEXT:   %malloccall4 = tail call noalias nonnull i8* @malloc(i32 %mallocsize3)
; CHECK-NEXT:   %cache.A5 = bitcast i8* %malloccall4 to double*
; CHECK:   %mul.i39 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %23 = icmp eq i32 %mul.i39, 0
; CHECK-NEXT:   br i1 %23, label %__enzyme_memcpy_double_mat_32.exit50, label %init.idx.i41

; CHECK: init.idx.i41:                                     ; preds = %init.end.i49, %__enzyme_memcpy_double_mat_32.exit38
; CHECK-NEXT:   %j.i40 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit38 ], [ %j.next.i48, %init.end.i49 ]
; CHECK-NEXT:   br label %for.body.i47

; CHECK: for.body.i47:                                     ; preds = %for.body.i47, %init.idx.i41
; CHECK-NEXT:   %i.i42 = phi i32 [ 0, %init.idx.i41 ], [ %i.next.i46, %for.body.i47 ]
; CHECK-NEXT:   %24 = mul nuw nsw i32 %j.i40, %N
; CHECK-NEXT:   %25 = add nuw nsw i32 %i.i42, %24
; CHECK-NEXT:   %dst.i.i43 = getelementptr inbounds double, double* %cache.A5, i32 %25
; CHECK-NEXT:   %26 = mul nuw nsw i32 %j.i40, %N
; CHECK-NEXT:   %27 = add nuw nsw i32 %i.i42, %26
; CHECK-NEXT:   %dst.i1.i44 = getelementptr inbounds double, double* %K, i32 %27
; CHECK-NEXT:   %src.i.l.i45 = load double, double* %dst.i1.i44, align 8
; CHECK-NEXT:   store double %src.i.l.i45, double* %dst.i.i43, align 8
; CHECK-NEXT:   %i.next.i46 = add nuw nsw i32 %i.i42, 1
; CHECK-NEXT:   %28 = icmp eq i32 %i.next.i46, %N
; CHECK-NEXT:   br i1 %28, label %init.end.i49, label %for.body.i47

; CHECK: init.end.i49:                                     ; preds = %for.body.i47
; CHECK-NEXT:   %j.next.i48 = add nuw nsw i32 %j.i40, 1
; CHECK-NEXT:   %29 = icmp eq i32 %j.next.i48, %N
; CHECK-NEXT:   br i1 %29, label %__enzyme_memcpy_double_mat_32.exit50, label %init.idx.i41

; CHECK: __enzyme_memcpy_double_mat_32.exit50:             ; preds = %__enzyme_memcpy_double_mat_32.exit38, %init.end.i49
; CHECK-NEXT:   %30 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   %mallocsize6 = mul nuw nsw i32 %30, 8
; CHECK-NEXT:   %malloccall7 = tail call noalias nonnull i8* @malloc(i32 %mallocsize6)
; CHECK-NEXT:   %cache.x8 = bitcast i8* %malloccall7 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %30, double* %x0, i32 1, double* %cache.x8, i32 1)
; CHECK-NEXT:   %31 = insertvalue { double*, double* } undef, double* %cache.A5, 0
; CHECK-NEXT:   %32 = insertvalue { double*, double* } %31, double* %cache.x8, 1
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %33 = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %33, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK:   %mul.i51 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %34 = icmp eq i32 %mul.i51, 0
; CHECK-NEXT:   br i1 %34, label %__enzyme_memcpy_double_mat_32.exit62, label %init.idx.i53

; CHECK: init.idx.i53:                                     ; preds = %init.end.i61, %__enzyme_memcpy_double_mat_32.exit50
; CHECK-NEXT:   %j.i52 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit50 ], [ %j.next.i60, %init.end.i61 ]
; CHECK-NEXT:   br label %for.body.i59

; CHECK: for.body.i59:                                     ; preds = %for.body.i59, %init.idx.i53
; CHECK-NEXT:   %i.i54 = phi i32 [ 0, %init.idx.i53 ], [ %i.next.i58, %for.body.i59 ]
; CHECK-NEXT:   %35 = mul nuw nsw i32 %j.i52, %N
; CHECK-NEXT:   %36 = add nuw nsw i32 %i.i54, %35
; CHECK-NEXT:   %dst.i.i55 = getelementptr inbounds double, double* %cache.A, i32 %36
; CHECK-NEXT:   %37 = mul nuw nsw i32 %j.i52, %N
; CHECK-NEXT:   %38 = add nuw nsw i32 %i.i54, %37
; CHECK-NEXT:   %dst.i1.i56 = getelementptr inbounds double, double* %K, i32 %38
; CHECK-NEXT:   %src.i.l.i57 = load double, double* %dst.i1.i56, align 8
; CHECK-NEXT:   store double %src.i.l.i57, double* %dst.i.i55, align 8
; CHECK-NEXT:   %i.next.i58 = add nuw nsw i32 %i.i54, 1
; CHECK-NEXT:   %39 = icmp eq i32 %i.next.i58, %N
; CHECK-NEXT:   br i1 %39, label %init.end.i61, label %for.body.i59

; CHECK: init.end.i61:                                     ; preds = %for.body.i59
; CHECK-NEXT:   %j.next.i60 = add nuw nsw i32 %j.i52, 1
; CHECK-NEXT:   %40 = icmp eq i32 %j.next.i60, %N
; CHECK-NEXT:   br i1 %40, label %__enzyme_memcpy_double_mat_32.exit62, label %init.idx.i53

; CHECK: __enzyme_memcpy_double_mat_32.exit62:             ; preds = %__enzyme_memcpy_double_mat_32.exit50, %init.end.i61
; CHECK-NEXT:   %41 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %41, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %41, double* %x0, i32 1, double* %cache.x, i32 1)
; CHECK-NEXT:   %42 = insertvalue { double*, double* } undef, double* %cache.A, 0
; CHECK-NEXT:   %43 = insertvalue { double*, double* } %42, double* %cache.x, 1
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %__enzyme_memcpy_double_mat_32.exit62
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %44 = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %45 = load double, double* %"x0'", align 8
; CHECK-NEXT:   %46 = fadd fast double %45, %44
; CHECK-NEXT:   store double %46, double* %"x0'", align 8
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %"v0'", i32 1, double* %x0, i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %K, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %47 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %47, double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %tape.ext.A = extractvalue { double*, double* } %43, 0
; CHECK-NEXT:   %tape.ext.x = extractvalue { double*, double* } %43, 1
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %"v0'", i32 1, double* %tape.ext.x, i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   %48 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %tape.ext.A, i32 %48, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %49 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %49, double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %50 = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %50)
; CHECK-NEXT:   %51 = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %51)
; CHECK-NEXT:   %tape.ext.A9 = extractvalue { double*, double* } %32, 0
; CHECK-NEXT:   %tape.ext.x10 = extractvalue { double*, double* } %32, 1
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %"v0'", i32 1, double* %tape.ext.x10, i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   %52 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %tape.ext.A9, i32 %52, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %53 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %53, double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %54 = bitcast double* %tape.ext.A9 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %54)
; CHECK-NEXT:   %55 = bitcast double* %tape.ext.x10 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %55)
; CHECK-NEXT:   %tape.ext.A17 = extractvalue { double*, double* } %21, 0
; CHECK-NEXT:   %tape.ext.x18 = extractvalue { double*, double* } %21, 1
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %"v0'", i32 1, double* %tape.ext.x18, i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   %56 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %tape.ext.A17, i32 %56, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %57 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %57, double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %58 = bitcast double* %tape.ext.A17 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %58)
; CHECK-NEXT:   %59 = bitcast double* %tape.ext.x18 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %59)
; CHECK-NEXT:   %tape.ext.A25 = extractvalue { double*, double* } %10, 0
; CHECK-NEXT:   %tape.ext.x26 = extractvalue { double*, double* } %10, 1
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %"v0'", i32 1, double* %tape.ext.x26, i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   %60 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %tape.ext.A25, i32 %60, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %61 = select i1 false, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %61, double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %62 = bitcast double* %tape.ext.A25 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %62)
; CHECK-NEXT:   %63 = bitcast double* %tape.ext.x26 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %63)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
