; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

; ModuleID = '/app/example.ll'
source_filename = "/app/example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [31 x i8] c";/app/example.c;compute;12;9;;\00", align 1
@1 = private unnamed_addr constant %0 { i32 0, i32 514, i32 0, i32 30, ptr @0 }, align 8
@2 = private unnamed_addr constant [33 x i8] c";/app/example.c;compute;12;106;;\00", align 1
@3 = private unnamed_addr constant %0 { i32 0, i32 514, i32 0, i32 32, ptr @2 }, align 8
@.gomp_critical_user_.reduction.var = common global [8 x i32] zeroinitializer
@4 = private unnamed_addr constant %0 { i32 0, i32 18, i32 0, i32 32, ptr @2 }, align 8
@5 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 30, ptr @0 }, align 8
@6 = private unnamed_addr constant [47 x i8] c"The sum of all array elements is equal to %f.\0A\00", align 1
@enzyme_dup = external local_unnamed_addr global i32, align 4
@enzyme_const = external local_unnamed_addr global i32, align 4
@7 = private unnamed_addr constant [5 x i8] c"%f, \00", align 1
@8 = private unnamed_addr constant [62 x i8] c"The derivative of the sum of the  array elements is equal to:\00", align 1
@9 = private unnamed_addr constant [59 x i8] c"Note last element is zero because we passed ARRAY_SIZE - 1\00", align 1
@10 = private unnamed_addr constant [37 x i8] c"Cannot allocate the array \22myArray\22.\00", align 1

; Function Attrs: nounwind uwtable
define dso_local double @compute(ptr noundef %arg, i32 noundef %arg1) #0 {
bb:
  %i = alloca ptr, align 8
  %i2 = alloca double, align 8
  store ptr %arg, ptr %i, align 8, !tbaa !8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %i2) #2
  store double 0.000000e+00, ptr %i2, align 8, !tbaa !12
  %i3 = zext i32 %arg1 to i64
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @5, i32 3, ptr nonnull @body, i64 %i3, ptr nonnull %i2, ptr nonnull %i)
  %i4 = load double, ptr %i2, align 8, !tbaa !12
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %i2) #2
  ret double %i4
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr #2

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define internal void @reduce(ptr nocapture noundef readonly %arg, ptr nocapture noundef readonly %arg1) #3 {
bb:
  %i = load ptr, ptr %arg1, align 8
  %i2 = load ptr, ptr %arg, align 8
  %i3 = load double, ptr %i2, align 8, !tbaa !12
  %i4 = load double, ptr %i, align 8, !tbaa !12
  %i5 = fadd double %i3, %i4
  store double %i5, ptr %i2, align 8, !tbaa !12
  ret void
}

; Function Attrs: convergent nounwind
declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr) local_unnamed_addr #4

; Function Attrs: convergent nounwind
declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr) local_unnamed_addr #4

; Function Attrs: alwaysinline norecurse nounwind uwtable
define internal void @body(ptr noalias nocapture noundef readonly %arg, ptr noalias nocapture readnone %arg1, i64 noundef %arg2, ptr nocapture noundef nonnull align 8 dereferenceable(8) %arg3, ptr nocapture noundef nonnull readonly align 8 dereferenceable(8) %arg4) #5 {
bb:
  %i = alloca i32, align 4
  %i5 = alloca i32, align 4
  %i6 = alloca i32, align 4
  %i7 = alloca i32, align 4
  %i8 = alloca double, align 8
  %i9 = alloca [1 x ptr], align 8
  %i10 = trunc i64 %arg2 to i32
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %i9)
  %i11 = icmp sgt i32 %i10, 0
  br i1 %i11, label %bb12, label %bb73

bb12:                                             ; preds = %bb
  %i13 = add nsw i32 %i10, -1
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i) #2, !noalias !14
  store i32 0, ptr %i, align 4, !tbaa !17, !noalias !14
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i5) #2, !noalias !14
  store i32 %i13, ptr %i5, align 4, !tbaa !17, !noalias !14
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i6) #2, !noalias !14
  store i32 1, ptr %i6, align 4, !tbaa !17, !noalias !14
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i7) #2, !noalias !14
  store i32 0, ptr %i7, align 4, !tbaa !17, !noalias !14
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %i8) #2, !noalias !14
  store double 0.000000e+00, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i14 = load i32, ptr %arg, align 4, !tbaa !17, !alias.scope !14
  call void @__kmpc_for_static_init_4(ptr nonnull @1, i32 %i14, i32 34, ptr nonnull %i7, ptr nonnull %i, ptr nonnull %i5, ptr nonnull %i6, i32 1, i32 1), !noalias !14
  %i15 = load i32, ptr %i5, align 4, !tbaa !17, !noalias !14
  %i16 = call i32 @llvm.smin.i32(i32 %i15, i32 %i13)
  store i32 %i16, ptr %i5, align 4, !tbaa !17, !noalias !14
  %i17 = load i32, ptr %i, align 4, !tbaa !17, !noalias !14
  %i18 = icmp slt i32 %i16, %i17
  br i1 %i18, label %bb63, label %bb19

bb19:                                             ; preds = %bb12
  %i20 = load ptr, ptr %arg4, align 8, !tbaa !8, !noalias !14
  %i21 = sext i32 %i17 to i64
  %i22 = add nsw i32 %i16, 1
  %i23 = add i32 %i16, 1
  %i24 = sub i32 %i23, %i17
  %i25 = sub i32 %i16, %i17
  %i26 = and i32 %i24, 3
  %i27 = icmp eq i32 %i26, 0
  br i1 %i27, label %bb38, label %bb28

bb28:                                             ; preds = %bb28, %bb19
  %i29 = phi double [ %i34, %bb28 ], [ 0.000000e+00, %bb19 ]
  %i30 = phi i64 [ %i35, %bb28 ], [ %i21, %bb19 ]
  %i31 = phi i32 [ %i36, %bb28 ], [ 0, %bb19 ]
  %i32 = getelementptr inbounds double, ptr %i20, i64 %i30
  %i33 = load double, ptr %i32, align 8, !tbaa !12, !noalias !14
  %i34 = fadd double %i29, %i33
  store double %i34, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i35 = add nsw i64 %i30, 1
  %i36 = add i32 %i31, 1
  %i37 = icmp eq i32 %i36, %i26
  br i1 %i37, label %bb38, label %bb28, !llvm.loop !19

bb38:                                             ; preds = %bb28, %bb19
  %i39 = phi double [ 0.000000e+00, %bb19 ], [ %i34, %bb28 ]
  %i40 = phi i64 [ %i21, %bb19 ], [ %i35, %bb28 ]
  %i41 = icmp ult i32 %i25, 3
  br i1 %i41, label %bb63, label %bb42

bb42:                                             ; preds = %bb42, %bb38
  %i43 = phi double [ %i59, %bb42 ], [ %i39, %bb38 ]
  %i44 = phi i64 [ %i60, %bb42 ], [ %i40, %bb38 ]
  %i45 = getelementptr inbounds double, ptr %i20, i64 %i44
  %i46 = load double, ptr %i45, align 8, !tbaa !12, !noalias !14
  %i47 = fadd double %i43, %i46
  store double %i47, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i48 = add nsw i64 %i44, 1
  %i49 = getelementptr inbounds double, ptr %i20, i64 %i48
  %i50 = load double, ptr %i49, align 8, !tbaa !12, !noalias !14
  %i51 = fadd double %i47, %i50
  store double %i51, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i52 = add nsw i64 %i44, 2
  %i53 = getelementptr inbounds double, ptr %i20, i64 %i52
  %i54 = load double, ptr %i53, align 8, !tbaa !12, !noalias !14
  %i55 = fadd double %i51, %i54
  store double %i55, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i56 = add nsw i64 %i44, 3
  %i57 = getelementptr inbounds double, ptr %i20, i64 %i56
  %i58 = load double, ptr %i57, align 8, !tbaa !12, !noalias !14
  %i59 = fadd double %i55, %i58
  store double %i59, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i60 = add nsw i64 %i44, 4
  %i61 = trunc i64 %i60 to i32
  %i62 = icmp eq i32 %i22, %i61
  br i1 %i62, label %bb63, label %bb42

bb63:                                             ; preds = %bb42, %bb38, %bb12
  call void @__kmpc_for_static_fini(ptr nonnull @3, i32 %i14), !noalias !14
  store ptr %i8, ptr %i9, align 8, !noalias !14
  %i64 = call i32 @__kmpc_reduce_nowait(ptr nonnull @4, i32 %i14, i32 1, i64 8, ptr nonnull %i9, ptr nonnull @reduce, ptr nonnull @.gomp_critical_user_.reduction.var), !noalias !14
  switch i32 %i64, label %bb72 [
    i32 1, label %bb65
    i32 2, label %bb69
  ]

bb65:                                             ; preds = %bb63
  %i66 = load double, ptr %arg3, align 8, !tbaa !12, !noalias !14
  %i67 = load double, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i68 = fadd double %i66, %i67
  store double %i68, ptr %arg3, align 8, !tbaa !12, !noalias !14
  call void @__kmpc_end_reduce_nowait(ptr nonnull @4, i32 %i14, ptr nonnull @.gomp_critical_user_.reduction.var), !noalias !14
  br label %bb72

bb69:                                             ; preds = %bb63
  %i70 = load double, ptr %i8, align 8, !tbaa !12, !noalias !14
  %i71 = atomicrmw fadd ptr %arg3, double %i70 monotonic, align 8, !noalias !14
  br label %bb72

bb72:                                             ; preds = %bb69, %bb65, %bb63
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %i8) #2, !noalias !14
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i7) #2, !noalias !14
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i6) #2, !noalias !14
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i5) #2, !noalias !14
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i) #2, !noalias !14
  br label %bb73

bb73:                                             ; preds = %bb72, %bb
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %i9)
  ret void
}

; Function Attrs: nounwind
declare !callback !21 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define void @caller(ptr %myArray, ptr %dmyArray) {
bb:
  %i15 = call double (...) @__enzyme_autodiff(ptr @compute, ptr @enzyme_const, ptr %myArray,  ptr @enzyme_const, i32 noundef 9) #2
  ret void
}

declare double @__enzyme_autodiff(...) local_unnamed_addr #9

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smin.i32(i32, i32) #11

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #13

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #2 = { nounwind }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { convergent nounwind }
attributes #5 = { alwaysinline norecurse nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc" "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { inaccessiblemem_or_argmemonly mustprogress nounwind willreturn allockind("free") "alloc-family"="malloc" "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #12 = { nofree nounwind }
attributes #13 = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }
attributes #14 = { inaccessiblememonly nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc" }
attributes #15 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!nvvm.annotations = !{}

!0 = !{i32 7, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"openmp", i32 50}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 4ba6a9c9f65bbc8bd06e3652cb20fd4dfc846137)"}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"double", !10, i64 0}
!14 = !{!15}
!15 = distinct !{!15, !16, !".omp_outlined._debug__: argument 0"}
!16 = distinct !{!16, !".omp_outlined._debug__"}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !10, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.unroll.disable"}
!21 = !{!22}
!22 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @diffebody(ptr noalias nocapture readonly %arg, ptr noalias nocapture readnone %arg1, i64 %arg2, ptr nocapture align 8 dereferenceable(8) %arg3, ptr nocapture align 8 %"arg3'", ptr nocapture readonly align 8 dereferenceable(8) %arg4, ptr nocapture align 8 %"arg4'", ptr %tapeArg) #13 {
; CHECK: bb:
; CHECK:   %truetape = load { ptr, ptr, ptr, ptr, ptr }, ptr %tapeArg, align 8, !enzyme_mustcache !61
; CHECK:   %0 = call i64 @omp_get_thread_num() #12
; CHECK:   %i_smpl = alloca i32, align 4
; CHECK:   %i5_smpl = alloca i32, align 4
; CHECK:   %i6_smpl = alloca i32, align 4
; CHECK:   %i7 = alloca i32, align 4
; CHECK:   %"{{.*}}'mi_fromtape" = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 0
; CHECK:   %1 = getelementptr inbounds ptr, ptr %"{{.*}}'mi_fromtape", i64 %0
; CHECK:   %"{{.*}}'mi" = load ptr, ptr %1, align 8
; CHECK:   %[[GEP2_BASE:.+]] = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 1
; CHECK:   %[[GEP2:.+]] = getelementptr inbounds ptr, ptr %[[GEP2_BASE]], i64 %0
; CHECK:   %{{.+}} = load ptr, ptr %[[GEP2]], align 8
; CHECK:   %i10 = trunc i64 %arg2 to i32
; CHECK:   %i11 = icmp sgt i32 %i10, 0
; CHECK:   br i1 %i11, label %bb12, label %invertbb73
;
; CHECK: bb12:                                             ; preds = %bb
; CHECK:   %i13 = add nsw i32 %i10, -1
; CHECK:   store i32 0, ptr %i7, align 4, !tbaa !17, !noalias !14
; CHECK:   %i14 = load i32, ptr %arg, align 4, !tbaa !17, !alias.scope !62, !noalias !65, !invariant.group !67
; CHECK:   store i32 0, ptr %i_smpl, align 4
; CHECK:   store i32 %i13, ptr %i5_smpl, align 4
; CHECK:   store i32 1, ptr %i6_smpl, align 4
; CHECK:   call void @__kmpc_for_static_init_4(ptr nonnull @1, i32 %i14, i32 34, ptr nonnull %i7, ptr nocapture nonnull %i_smpl, ptr nocapture nonnull %i5_smpl, ptr nocapture nonnull %i6_smpl, i32 1, i32 1) #15, !noalias !14
; CHECK:   %3 = load i32, ptr %i5_smpl, align 4, !invariant.group !68
; CHECK:   %4 = load i32, ptr %i_smpl, align 4, !invariant.group !69
; CHECK:   %i16 = call i32 @llvm.smin.i32(i32 %3, i32 %i13) #15
; CHECK:   %i18 = icmp slt i32 %i16, %4
; CHECK:   br i1 %i18, label %bb63, label %bb19
;
; CHECK: bb19:                                             ; preds = %bb12
; CHECK:   %"i20'il_phi_fromtape" = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 2
; CHECK:   %5 = getelementptr inbounds ptr, ptr %"i20'il_phi_fromtape", i64 %0
; CHECK:   %"i20'il_phi" = load ptr, ptr %5, align 8
; CHECK:   %i22 = add i32 %i16, 1
; CHECK:   %i24 = sub i32 %i22, %4
; CHECK:   %i25 = sub i32 %i16, %4
; CHECK:   %i26 = and i32 %i24, 3
; CHECK:   %i27 = icmp eq i32 %i26, 0
; CHECK:   br i1 %i27, label %bb38, label %bb28
;
; CHECK: bb28:                                             ; preds = %bb19, %bb28
; CHECK:   %iv = phi i64 [ %iv.next, %bb28 ], [ 0, %bb19 ]
; CHECK:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK:   %6 = trunc i64 %iv to i32
; CHECK:   %i36 = add i32 %6, 1
; CHECK:   %i37 = icmp eq i32 %i36, %i26
; CHECK:   br i1 %i37, label %bb38, label %bb28, !llvm.loop !19
;
; CHECK: bb38:                                             ; preds = %bb28, %bb19
; CHECK:   %i40_fromtape = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 4
; CHECK:   %7 = getelementptr inbounds i64, ptr %i40_fromtape, i64 %0
; CHECK:   %i40 = load i64, ptr %7, align 8, !invariant.group !70
; CHECK:   %i41 = icmp ult i32 %i25, 3
; CHECK:   br i1 %i41, label %bb63, label %bb42
;
; CHECK: bb42:                                             ; preds = %bb38, %bb42
; CHECK:   %[[IV1:iv[0-9]*]] = phi i64 [ %[[IV_NEXT:iv.next[0-9]*]], %bb42 ], [ 0, %bb38 ]
; CHECK:   %[[IV_NEXT]] = add {{(nuw )?}}{{(nsw )?}}i64 %[[IV1]], 1
; CHECK:   %{{[0-9]+}} = shl {{(nuw )?}}{{(nsw )?}}i64 %[[IV1]], 2
; CHECK:   %9 = add nsw i64 %i40, %8
; CHECK:   %i60 = add nsw i64 %9, 4
; CHECK:   %i61 = trunc i64 %i60 to i32
; CHECK:   %i62 = icmp eq i32 %i22, %i61
; CHECK:   br i1 %i62, label %bb63, label %bb42
;
; CHECK: bb63:                                             ; preds = %bb42, %bb38, %bb12
; CHECK:   %_cache.0 = phi i8 [ {{[0-9]}}, %bb{{.*}} ], [ {{[0-9]}}, %bb{{.*}} ], [ {{[0-9]}}, %bb{{.*}} ]
; CHECK:   %"i20'il_phi_cache.0" = phi ptr [ undef, %bb12 ], [ %"i20'il_phi", %bb38 ], [ %"i20'il_phi", %bb42 ]
; CHECK:   %i64_fromtape = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 3
; CHECK:   %10 = getelementptr inbounds i32, ptr %i64_fromtape, i64 %0
; CHECK:   %i64 = load i32, ptr %10, align 4
; CHECK:   br label %invertbb73
;
; CHECK: invertbb:                                         ; preds = %invertbb73, %invertbb12
; CHECK:   call void @free(ptr nonnull %"{{.*}}'mi")
; CHECK:   call void @free(ptr %{{.*}})
; CHECK:   ret void
;
; CHECK: invertbb12:                                       ; preds = %invertbb28, %invertbb38, %invertbb63
; CHECK:   %i14_unwrap = load i32, ptr %arg, align 4, !tbaa !17, !alias.scope !62, !noalias !65, !invariant.group !67
; CHECK:   call void @__kmpc_for_static_fini(ptr @1, i32 %i14_unwrap)
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   br label %invertbb
;
; CHECK: invertbb28:                                       ; preds = %invertbb38.loopexit, %incinvertbb28
; CHECK:   %"i34'de.0" = phi double [ %18, %invertbb38.loopexit ], [ %15, %incinvertbb28 ]
; CHECK:   %"iv'ac.0" = phi i64 [ %_unwrap5, %invertbb38.loopexit ], [ %16, %incinvertbb28 ]
; CHECK:   %11 = load double, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   %12 = fadd fast double %"i34'de.0", %11
; CHECK:   %_unwrap = load i32, ptr %i_smpl, align 4, !alias.scope !76, !noalias !79, !invariant.group !69
; CHECK:   %i21_unwrap = sext i32 %_unwrap to i64
; CHECK:   %_unwrap1 = add nsw i64 %i21_unwrap, %"iv'ac.0"
; CHECK:   %"i32'ipg_unwrap" = getelementptr inbounds double, ptr %"i20'il_phi_cache.1", i64 %_unwrap1
; CHECK:   %13 = atomicrmw fadd ptr %"i32'ipg_unwrap", double %12 monotonic, align 8
; CHECK:   %14 = icmp eq i64 %"iv'ac.0", 0
; CHECK:   %15 = select fast i1 %14, double 0.000000e+00, double %12
; CHECK:   br i1 %14, label %invertbb12, label %incinvertbb28
;
; CHECK: incinvertbb28:                                    ; preds = %invertbb28
; CHECK:   %16 = add nsw i64 %"iv'ac.0", -1
; CHECK:   br label %invertbb28
;
; CHECK: invertbb38.loopexit:                              ; preds = %invertbb38
; CHECK:   %_unwrap4 = add nsw i32 %i26_unwrap11, -1
; CHECK:   %_unwrap5 = zext i32 %_unwrap4 to i64
; CHECK:   br label %invertbb28
;
; CHECK: invertbb38:                                       ; preds = %invertbb42, %invertbb63
; CHECK:   %"i39'de.0" = phi double [ 0.000000e+00, %invertbb63 ], [ %33, %invertbb42 ]
; CHECK:   %_unwrap6 = load i32, ptr %i5_smpl, align 4, !alias.scope !81, !noalias !84, !invariant.group !68
; CHECK:   %i13_unwrap7 = add nsw i32 %i10, -1
; CHECK:   %17 = call i32 @llvm.smin.i32(i32 %_unwrap6, i32 %i13_unwrap7) #15
; CHECK:   %i23_unwrap8 = add i32 %17, 1
; CHECK:   %_unwrap9 = load i32, ptr %i_smpl, align 4, !alias.scope !76, !noalias !79, !invariant.group !69
; CHECK:   %i24_unwrap10 = sub i32 %i23_unwrap8, %_unwrap9
; CHECK:   %i26_unwrap11 = and i32 %i24_unwrap10, 3
; CHECK:   %i27_unwrap = icmp eq i32 %i26_unwrap11, 0
; CHECK:   %18 = select fast i1 %i27_unwrap, double 0.000000e+00, double %"i39'de.0"
; CHECK:   br i1 %i27_unwrap, label %invertbb12, label %invertbb38.loopexit
;
; CHECK: invertbb42:                                       ; preds = %invertbb63.loopexit, %incinvertbb42
; CHECK:   %"i59'de.0" = phi double [ 0.000000e+00, %invertbb63.loopexit ], [ %34, %incinvertbb42 ]
; CHECK:   %"i39'de.1" = phi double [ 0.000000e+00, %invertbb63.loopexit ], [ %33, %incinvertbb42 ]
; CHECK:   %"[[IV1]]'ac.0" = phi i64 [ %_unwrap21, %invertbb63.loopexit ], [ %35, %incinvertbb42 ]
; CHECK:   %19 = load double, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   %20 = fadd fast double %"i59'de.0", %19
; CHECK:   %{{[0-9a-zA-Z_]+}} = load i64, ptr %{{[0-9a-zA-Z_]+}}, align 8, !alias.scope !86, !noalias !89, !invariant.group !70
; CHECK:   %_unwrap12 = shl {{(nuw )?}}{{(nsw )?}}i64 %"[[IV1]]'ac.0", 2
; CHECK:   %_unwrap13 = add nsw i64 %{{[0-9a-zA-Z_]+}}, %_unwrap12
; CHECK:   %i56_unwrap = add nsw i64 %_unwrap13, 3
; CHECK:   %"i57'ipg_unwrap" = getelementptr inbounds double, ptr %"i20'il_phi_cache.1", i64 %i56_unwrap
; CHECK:   %21 = atomicrmw fadd ptr %"i57'ipg_unwrap", double %20 monotonic, align 8
; CHECK:   %22 = load double, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   %23 = fadd fast double %20, %22
; CHECK:   %i52_unwrap = add nsw i64 %_unwrap13, 2
; CHECK:   %"i53'ipg_unwrap" = getelementptr inbounds double, ptr %"i20'il_phi_cache.1", i64 %i52_unwrap
; CHECK:   %24 = atomicrmw fadd ptr %"i53'ipg_unwrap", double %23 monotonic, align 8
; CHECK:   %25 = load double, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   %26 = fadd fast double %23, %25
; CHECK:   %i48_unwrap = add nsw i64 %_unwrap13, 1
; CHECK:   %"i49'ipg_unwrap" = getelementptr inbounds double, ptr %"i20'il_phi_cache.1", i64 %i48_unwrap
; CHECK:   %27 = atomicrmw fadd ptr %"i49'ipg_unwrap", double %26 monotonic, align 8
; CHECK:   %28 = load double, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   store double 0.000000e+00, ptr %"{{.*}}'mi", align 8, !tbaa !12, !alias.scope !71, !noalias !74
; CHECK:   %29 = fadd fast double %26, %28
; CHECK:   %"i45'ipg_unwrap" = getelementptr inbounds double, ptr %"i20'il_phi_cache.1", i64 %_unwrap13
; CHECK:   %30 = atomicrmw fadd ptr %"i45'ipg_unwrap", double %29 monotonic, align 8
; CHECK:   %31 = icmp eq i64 %"[[IV1]]'ac.0", 0
; CHECK:   %32 = fadd fast double %"i39'de.1", %29
; CHECK:   %33 = select fast i1 %31, double %32, double %"i39'de.1"
; CHECK:   %34 = select fast i1 %31, double 0.000000e+00, double %29
; CHECK:   br i1 %31, label %invertbb38, label %incinvertbb42
;
; CHECK: incinvertbb42:                                    ; preds = %invertbb42
; CHECK:   %35 = add nsw i64 %"[[IV1]]'ac.0", -1
; CHECK:   br label %invertbb42
;
; CHECK: invertbb63.loopexit:                              ; preds = %invertbb63
; CHECK:   %_unwrap15 = load i32, ptr %i5_smpl, align 4, !alias.scope !81, !noalias !84, !invariant.group !68
; CHECK:   %i13_unwrap16 = add nsw i32 %i10, -1
; CHECK:   %36 = call i32 @llvm.smin.i32(i32 %_unwrap15, i32 %i13_unwrap16) #15
; CHECK:   %_unwrap17 = add i32 %36, -3
; CHECK:   %{{[a-zA-Z0-9_]+}} = extractvalue { ptr, ptr, ptr, ptr, ptr } %truetape, 4
; CHECK:   %{{[a-zA-Z0-9_]+}} = getelementptr inbounds i64, ptr %{{[a-zA-Z0-9_]+}}, i64 %0
; CHECK:   %{{[a-zA-Z0-9_]+}} = load i64, ptr %{{[a-zA-Z0-9_]+}}, align 8, !alias.scope !86, !noalias !89, !invariant.group !70
; CHECK:   %{{[a-zA-Z0-9_]+}} = trunc i64 %{{[a-zA-Z0-9_]+}} to i32
; CHECK:   %{{[a-zA-Z0-9_]+}} = sub i32 %_unwrap17, %{{[a-zA-Z0-9_]+}}
; CHECK:   %{{[a-zA-Z0-9_]+}} = zext i32 %{{[a-zA-Z0-9_]+}} to i64
; CHECK:   %_unwrap21 = lshr i64 %{{[a-zA-Z0-9_]+}}, 2
; CHECK:   br label %invertbb42
;
; CHECK:   switch i8 %_cache.1, label %invertbb{{.*}} [
; CHECK-DAG:     i8 {{[0-9]}}, label %invertbb{{.*}}
; CHECK-DAG:     i8 {{[0-9]}}, label %invertbb{{.*}}
; CHECK:   ]
;
; CHECK: invertbb65:                                       ; preds = %invertbb72
; CHECK:   %37 = load double, ptr %"arg3'", align 8, !tbaa !12, !alias.scope !91, !noalias !94
; CHECK:   store double 0.000000e+00, ptr %"arg3'", align 8, !tbaa !12, !alias.scope !91, !noalias !94
; CHECK:   %38 = atomicrmw fadd ptr %"{{.*}}'mi", double %37 monotonic, align 8
; CHECK:   %39 = atomicrmw fadd ptr %"arg3'", double %37 monotonic, align 8
; CHECK:   br label %invertbb63
;
; CHECK: invertbb69:                                       ; preds = %invertbb72
; CHECK:   %40 = load atomic double, ptr %"arg3'" monotonic, align 8
; CHECK:   %41 = atomicrmw fadd ptr %"{{.*}}'mi", double %40 monotonic, align 8
; CHECK:   br label %invertbb63
;
; CHECK: invertbb72:                                       ; preds = %invertbb73
; CHECK:   switch i32 %i64_cache.0, label %invertbb63 [
; CHECK:     i32 1, label %invertbb65
; CHECK:     i32 2, label %invertbb69
; CHECK:   ]
;
; CHECK: invertbb73:                                       ; preds = %bb, %bb63
; CHECK:   %i64_cache.0 = phi i32 [ %i64, %bb63 ], [ undef, %bb ]
; CHECK:   %_cache.1 = phi i8 [ %_cache.0, %bb63 ], [ undef, %bb ]
; CHECK:   %"i20'il_phi_cache.1" = phi ptr [ %"i20'il_phi_cache.0", %bb63 ], [ undef, %bb ]
; CHECK:   br i1 %i11, label %invertbb72, label %invertbb
; CHECK: }
