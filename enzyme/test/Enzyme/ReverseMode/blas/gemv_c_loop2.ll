;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S | FileCheck %s


; Function Attrs: nounwind uwtable
define double @coupled_springs(double* noundef %K, double* nocapture readnone %m, double* noundef %x0, double* noundef %v0, double %T, i32 noundef %N) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %0 = load double, double* %x0, align 8
  ret double %0

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1) #4
  %inc = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %inc, 5000
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
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
; CHECK-NEXT:   %"iv'ac" = alloca i64, align 8
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %malloccall2_cache = alloca i8**, align 8
; CHECK-NEXT:   %malloccall_cache = alloca i8**, align 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull dereferenceable(40000) dereferenceable_or_null(40000) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 40000), !enzyme_cache_alloc !0
; CHECK-NEXT:   %malloccall2_malloccache = bitcast i8* %malloccall6 to i8**
; CHECK-NEXT:   store i8** %malloccall2_malloccache, i8*** %malloccall2_cache, align 8, !invariant.group !2
; CHECK-NEXT:   %malloccall16 = tail call noalias nonnull dereferenceable(40000) dereferenceable_or_null(40000) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 40000), !enzyme_cache_alloc !3
; CHECK-NEXT:   %malloccall_malloccache = bitcast i8* %malloccall16 to i8**
; CHECK-NEXT:   store i8** %malloccall_malloccache, i8*** %malloccall_cache, align 8, !invariant.group !5
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body
; CHECK-NEXT:   br label %invertfor.cond.cleanup

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = mul i32 %N, %N
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @cblas_dlacpy(i32 101, i8 0, i32 %N, i32 %N, double* %K, i32 %N, double* %cache.A, i32 %N)
; CHECK-NEXT:   %1 = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %1, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %2 = load i8**, i8*** %malloccall2_cache, align 8, !dereferenceable !6, !invariant.group !2
; CHECK-NEXT:   %3 = getelementptr inbounds i8*, i8** %2, i64 %iv
; CHECK-NEXT:   store i8* %malloccall2, i8** %3, align 8, !invariant.group !7
; CHECK-NEXT:   %4 = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !6, !invariant.group !5
; CHECK-NEXT:   %5 = getelementptr inbounds i8*, i8** %4, i64 %iv
; CHECK-NEXT:   store i8* %malloccall, i8** %5, align 8, !invariant.group !8
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %1, double* %x0, i32 1, double* %cache.x, i32 1)
; CHECK-NEXT:   tail call void @cblas_dgemv(i32 noundef 101, i32 noundef 111, i32 noundef %N, i32 noundef %N, double noundef 1.000000e-03, double* noundef %K, i32 noundef %N, double* noundef %x0, i32 noundef 1, double noundef 1.000000e+00, double* noundef %v0, i32 noundef 1)
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %iv.next, 5000
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %forfree = load i8**, i8*** %malloccall2_cache, align 8, !dereferenceable !6, !invariant.group !2
; CHECK-NEXT:   %[[i9:.+]] = bitcast i8** %forfree to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i9]]), !enzyme_cache_free !0
; CHECK-NEXT:   %[[i10:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %forfree17 = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !6, !invariant.group !5
; CHECK-NEXT:   %[[i11:.+]] = bitcast i8** %forfree17 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i11]]), !enzyme_cache_free !3
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond.cleanup:                           ; preds = %for.cond.cleanup
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %[[i13:.+]] = load double, double* %"x0'", align 8, !alias.scope !9, !noalias !12
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i13]], %[[i12]]
; CHECK-NEXT:   store double %[[i14]], double* %"x0'", align 8, !alias.scope !9, !noalias !12
; CHECK-NEXT:   br label %mergeinvertfor.body_for.cond.cleanup

; CHECK: mergeinvertfor.body_for.cond.cleanup:             ; preds = %invertfor.cond.cleanup
; CHECK-NEXT:   store i64 4999, i64* %"iv'ac", align 4
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %mergeinvertfor.body_for.cond.cleanup
; CHECK-NEXT:   %[[i15:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i16:.+]] = load i8**, i8*** %malloccall2_cache, align 8, !dereferenceable !6, !invariant.group !2
; CHECK-NEXT:   %[[i17:.+]] = getelementptr inbounds i8*, i8** %[[i16]], i64 %[[i15]]
; CHECK-NEXT:   %[[i18:.+]] = load i8*, i8** %[[i17]], align 8, !invariant.group !7
; CHECK-NEXT:   %cache.x_unwrap = bitcast i8* %[[i18]] to double*
; CHECK-NEXT:   %[[i19:.+]] = load i8**, i8*** %malloccall_cache, align 8, !dereferenceable !6, !invariant.group !5
; CHECK-NEXT:   %[[i20:.+]] = getelementptr inbounds i8*, i8** %[[i19]], i64 %[[i15]]
; CHECK-NEXT:   %[[i21:.+]] = load i8*, i8** %[[i20]], align 8, !invariant.group !8
; CHECK-NEXT:   %cache.A_unwrap = bitcast i8* %[[i21]] to double*
; CHECK-DAG:   %[[r20:.+]] = select i1 true, double* %"v0'", double* %cache.x_unwrap
; CHECK-DAG:   %[[r21:.+]] = select i1 true, double* %cache.x_unwrap, double* %"v0'"
; CHECK-NEXT:   call void @cblas_dger(i32 101, i32 %N, i32 %N, double 1.000000e-03, double* %[[r20]], i32 1, double* %[[r21]], i32 1, double* %"K'", i32 %N)
; CHECK-NEXT:   call void @cblas_dgemv(i32 101, i32 112, i32 %N, i32 %N, double 1.000000e-03, double* %cache.A_unwrap, i32 %N, double* %"v0'", i32 1, double 1.000000e+00, double* %"x0'", i32 1)
; CHECK-NEXT:   %[[i23:.+]] = select i1 true, i32 %N, i32 %N
; CHECK-NEXT:   call void @cblas_dscal(i32 %[[i23]], double 1.000000e+00, double* %"v0'", i32 1)
; CHECK-NEXT:   %[[i24:.+]] = bitcast double* %cache.A_unwrap to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i24]])
; CHECK-NEXT:   %[[i25:.+]] = bitcast double* %cache.x_unwrap to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i25]])
; CHECK-NEXT:   %[[i26:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i27:.+]] = icmp eq i64 %[[i26]], 0
; CHECK-NEXT:   %[[i28:.+]] = xor i1 %[[i27]], true
; CHECK-NEXT:   br i1 %[[i27]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[i29:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i30:.+]] = add nsw i64 %[[i29]], -1
; CHECK-NEXT:   store i64 %[[i30]], i64* %"iv'ac", align 4
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
