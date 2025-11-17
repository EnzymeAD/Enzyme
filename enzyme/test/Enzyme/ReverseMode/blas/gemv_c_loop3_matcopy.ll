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

; CHECK: define internal void @diffecoupled_springs(double* nocapture noundef readonly %K, double* nocapture %"K'", double* nocapture readnone %m, double* nocapture noundef readonly %x0, double* nocapture %"x0'", double* nocapture noundef %v0, double* nocapture %"v0'", double %T, i32 noundef %N, double %differeturn)
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
; CHECK-NEXT:   %mallocsize22 = mul nuw nsw i32 %N, 8
; CHECK-NEXT:   %malloccall23 = tail call noalias nonnull i8* @malloc(i32 %mallocsize22)
; CHECK-NEXT:   %cache.x24 = bitcast i8* %malloccall23 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %N, double* %x0, i32 1, double* %cache.x24, i32 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %[[i11:.+]] = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize11 = mul nuw nsw i32 %[[i11]], 8
; CHECK-NEXT:   %malloccall12 = tail call noalias nonnull i8* @malloc(i32 %mallocsize11)
; CHECK-NEXT:   %cache.A13 = bitcast i8* %malloccall12 to double*
; CHECK:   %mul.i27 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %[[i12:.+]] = icmp eq i32 %mul.i27, 0
; CHECK-NEXT:   br i1 %[[i12]], label %__enzyme_memcpy_double_mat_32.exit38, label %init.idx.i29

; CHECK: init.idx.i29:                                     ; preds = %init.end.i37, %__enzyme_memcpy_double_mat_32.exit
; CHECK-NEXT:   %j.i28 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit ], [ %j.next.i36, %init.end.i37 ]
; CHECK-NEXT:   br label %for.body.i35

; CHECK: for.body.i35:                                     ; preds = %for.body.i35, %init.idx.i29
; CHECK-NEXT:   %i.i30 = phi i32 [ 0, %init.idx.i29 ], [ %i.next.i34, %for.body.i35 ]
; CHECK-NEXT:   %[[i13:.+]] = mul nuw nsw i32 %j.i28, %N
; CHECK-NEXT:   %[[i14:.+]] = add nuw nsw i32 %i.i30, %[[i13]]
; CHECK-NEXT:   %dst.i.i31 = getelementptr inbounds double, double* %cache.A13, i32 %[[i14]]
; CHECK-NEXT:   %[[i15:.+]] = mul nuw nsw i32 %j.i28, %N
; CHECK-NEXT:   %[[i16:.+]] = add nuw nsw i32 %i.i30, %[[i15]]
; CHECK-NEXT:   %dst.i1.i32 = getelementptr inbounds double, double* %K, i32 %[[i16]]
; CHECK-NEXT:   %src.i.l.i33 = load double, double* %dst.i1.i32, align 8
; CHECK-NEXT:   store double %src.i.l.i33, double* %dst.i.i31, align 8
; CHECK-NEXT:   %i.next.i34 = add nuw nsw i32 %i.i30, 1
; CHECK-NEXT:   %[[i17:.+]] = icmp eq i32 %i.next.i34, %N
; CHECK-NEXT:   br i1 %[[i17]], label %init.end.i37, label %for.body.i35

; CHECK: init.end.i37:                                     ; preds = %for.body.i35
; CHECK-NEXT:   %j.next.i36 = add nuw nsw i32 %j.i28, 1
; CHECK-NEXT:   %[[i18:.+]] = icmp eq i32 %j.next.i36, %N
; CHECK-NEXT:   br i1 %[[i18:.+]], label %__enzyme_memcpy_double_mat_32.exit38, label %init.idx.i29

; CHECK: __enzyme_memcpy_double_mat_32.exit38:             ; preds = %__enzyme_memcpy_double_mat_32.exit, %init.end.i37
; CHECK-NEXT:   %mallocsize14 = mul nuw nsw i32 %N, 8
; CHECK-NEXT:   %malloccall15 = tail call noalias nonnull i8* @malloc(i32 %mallocsize14)
; CHECK-NEXT:   %cache.x16 = bitcast i8* %malloccall15 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %N, double* %x0, i32 1, double* %cache.x16, i32 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %[[i22:.+]] = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize3 = mul nuw nsw i32 %[[i22]], 8
; CHECK-NEXT:   %malloccall4 = tail call noalias nonnull i8* @malloc(i32 %mallocsize3)
; CHECK-NEXT:   %cache.A5 = bitcast i8* %malloccall4 to double*
; CHECK:   %mul.i39 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %[[i23:.+]] = icmp eq i32 %mul.i39, 0
; CHECK-NEXT:   br i1 %[[i23]], label %__enzyme_memcpy_double_mat_32.exit50, label %init.idx.i41

; CHECK: init.idx.i41:                                     ; preds = %init.end.i49, %__enzyme_memcpy_double_mat_32.exit38
; CHECK-NEXT:   %j.i40 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit38 ], [ %j.next.i48, %init.end.i49 ]
; CHECK-NEXT:   br label %for.body.i47

; CHECK: for.body.i47:                                     ; preds = %for.body.i47, %init.idx.i41
; CHECK-NEXT:   %i.i42 = phi i32 [ 0, %init.idx.i41 ], [ %i.next.i46, %for.body.i47 ]
; CHECK-NEXT:   %[[i24:.+]] = mul nuw nsw i32 %j.i40, %N
; CHECK-NEXT:   %[[i25:.+]] = add nuw nsw i32 %i.i42, %[[i24]]
; CHECK-NEXT:   %dst.i.i43 = getelementptr inbounds double, double* %cache.A5, i32 %[[i25]]
; CHECK-NEXT:   %[[i26:.+]] = mul nuw nsw i32 %j.i40, %N
; CHECK-NEXT:   %[[i27:.+]] = add nuw nsw i32 %i.i42, %[[i26]]
; CHECK-NEXT:   %dst.i1.i44 = getelementptr inbounds double, double* %K, i32 %[[i27]]
; CHECK-NEXT:   %src.i.l.i45 = load double, double* %dst.i1.i44, align 8
; CHECK-NEXT:   store double %src.i.l.i45, double* %dst.i.i43, align 8
; CHECK-NEXT:   %i.next.i46 = add nuw nsw i32 %i.i42, 1
; CHECK-NEXT:   %[[i28:.+]] = icmp eq i32 %i.next.i46, %N
; CHECK-NEXT:   br i1 %[[i28]], label %init.end.i49, label %for.body.i47

; CHECK: init.end.i49:                                     ; preds = %for.body.i47
; CHECK-NEXT:   %j.next.i48 = add nuw nsw i32 %j.i40, 1
; CHECK-NEXT:   %[[i29:.+]] = icmp eq i32 %j.next.i48, %N
; CHECK-NEXT:   br i1 %[[i29]], label %__enzyme_memcpy_double_mat_32.exit50, label %init.idx.i41

; CHECK: __enzyme_memcpy_double_mat_32.exit50:             ; preds = %__enzyme_memcpy_double_mat_32.exit38, %init.end.i49
; CHECK-NEXT:   %mallocsize6 = mul nuw nsw i32 %N, 8
; CHECK-NEXT:   %malloccall7 = tail call noalias nonnull i8* @malloc(i32 %mallocsize6)
; CHECK-NEXT:   %cache.x8 = bitcast i8* %malloccall7 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %N, double* %x0, i32 1, double* %cache.x8, i32 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %[[i33:.+]] = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %[[i33]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK:   %mul.i51 = add nuw nsw i32 %N, %N
; CHECK-NEXT:   %[[i34:.+]] = icmp eq i32 %mul.i51, 0
; CHECK-NEXT:   br i1 %[[i34]], label %__enzyme_memcpy_double_mat_32.exit62, label %init.idx.i53

; CHECK: init.idx.i53:                                     ; preds = %init.end.i61, %__enzyme_memcpy_double_mat_32.exit50
; CHECK-NEXT:   %j.i52 = phi i32 [ 0, %__enzyme_memcpy_double_mat_32.exit50 ], [ %j.next.i60, %init.end.i61 ]
; CHECK-NEXT:   br label %for.body.i59

; CHECK: for.body.i59:                                     ; preds = %for.body.i59, %init.idx.i53
; CHECK-NEXT:   %i.i54 = phi i32 [ 0, %init.idx.i53 ], [ %i.next.i58, %for.body.i59 ]
; CHECK-NEXT:   %[[i35:.+]] = mul nuw nsw i32 %j.i52, %N
; CHECK-NEXT:   %[[i36:.+]] = add nuw nsw i32 %i.i54, %[[i35]]
; CHECK-NEXT:   %dst.i.i55 = getelementptr inbounds double, double* %cache.A, i32 %[[i36]]
; CHECK-NEXT:   %[[i37:.+]] = mul nuw nsw i32 %j.i52, %N
; CHECK-NEXT:   %[[i38:.+]] = add nuw nsw i32 %i.i54, %[[i37]]
; CHECK-NEXT:   %dst.i1.i56 = getelementptr inbounds double, double* %K, i32 %[[i38]]
; CHECK-NEXT:   %src.i.l.i57 = load double, double* %dst.i1.i56, align 8
; CHECK-NEXT:   store double %src.i.l.i57, double* %dst.i.i55, align 8
; CHECK-NEXT:   %i.next.i58 = add nuw nsw i32 %i.i54, 1
; CHECK-NEXT:   %[[i39:.+]] = icmp eq i32 %i.next.i58, %N
; CHECK-NEXT:   br i1 %[[i39]], label %init.end.i61, label %for.body.i59

; CHECK: init.end.i61:                                     ; preds = %for.body.i59
; CHECK-NEXT:   %j.next.i60 = add nuw nsw i32 %j.i52, 1
; CHECK-NEXT:   %[[i40:.+]] = icmp eq i32 %j.next.i60, %N
; CHECK-NEXT:   br i1 %[[i40]], label %__enzyme_memcpy_double_mat_32.exit62, label %init.idx.i53

; CHECK: __enzyme_memcpy_double_mat_32.exit62:             ; preds = %__enzyme_memcpy_double_mat_32.exit50, %init.end.i61
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %N, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %N, double* %x0, i32 1, double* %cache.x, i32 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %__enzyme_memcpy_double_mat_32.exit62
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %[[i44:.+]] = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %[[i45:.+]] = load double, double* %"x0'", align 8
; CHECK-NEXT:   %[[i46:.+]] = fadd fast double %[[i45:.+]], %[[i44]]
; CHECK-NEXT:   store double %[[i46:.+]], double* %"x0'", align 8
; CHECK-DAG:   %[[r39:.+]] = select i1 true, double* %"v0'", double* %x0
; CHECK-DAG:   %[[r40:.+]] = select i1 true, double* %x0, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r39]], i32 1, double* %[[r40]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %K, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i47:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i47]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-DAG:   %[[r42:.+]] = select i1 true, double* %"v0'", double* %cache.x
; CHECK-DAG:   %[[r43:.+]] = select i1 true, double* %cache.x, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r42]], i32 1, double* %[[r43]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %cache.A, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i49:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i49]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %[[i50:.+]] = bitcast double* %cache.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i50]])
; CHECK-NEXT:   %[[i51:.+]] = bitcast double* %cache.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i51]])
; CHECK-DAG:   %[[r48:.+]] = select i1 true, double* %"v0'", double* %cache.x8
; CHECK-DAG:   %[[r49:.+]] = select i1 true, double* %cache.x8, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r48]], i32 1, double* %[[r49]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %cache.A5, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i53:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i53]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %[[i54:.+]] = bitcast double* %cache.A5 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i54]])
; CHECK-NEXT:   %[[i55:.+]] = bitcast double* %cache.x8 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i55]])
; CHECK-DAG:   %[[r54:.+]] = select i1 true, double* %"v0'", double* %cache.x16
; CHECK-DAG:   %[[r55:.+]] = select i1 true, double* %cache.x16, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r54]], i32 1, double* %[[r55]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %cache.A13, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i57:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i57]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %[[i58:.+]] = bitcast double* %cache.A13 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i58]])
; CHECK-NEXT:   %[[i59:.+]] = bitcast double* %cache.x16 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i59]])
; CHECK-DAG:   %[[r60:.+]] = select i1 true, double* %"v0'", double* %cache.x24
; CHECK-DAG:   %[[r61:.+]] = select i1 true, double* %cache.x24, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r60]], i32 1, double* %[[r61]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %cache.A21, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i61:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i61]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %[[i62:.+]] = bitcast double* %cache.A21 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i62]])
; CHECK-NEXT:   %[[i63:.+]] = bitcast double* %cache.x24 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i63]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
