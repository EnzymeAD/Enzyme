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

;CHECK: define internal void @diffebody(ptr noalias nocapture readonly %arg, ptr noalias nocapture readnone %arg1, i64 %arg2, ptr nocapture align 8 dereferenceable(8) %arg3, ptr nocapture align 8 %"arg3'", ptr nocapture readonly align 8 dereferenceable(8) %arg4, ptr %tapeArg)
;CHECK-NEXT: bb:
;CHECK-NEXT:   %truetape = load { ptr, ptr, ptr }, ptr %tapeArg, align 8i
;CHECK-NEXT:   %0 = call i64 @omp_get_thread_num()
;CHECK-NEXT:   %i_smpl = alloca i32, align 4
;CHECK-NEXT:   %i5_smpl = alloca i32, align 4
;CHECK-NEXT:   %i6_smpl = alloca i32, align 4
;CHECK-NEXT:   %i7 = alloca i32, align 4
;CHECK-NEXT:   %malloccall_fromtape = extractvalue { ptr, ptr, ptr } %truetape, 0
;CHECK-NEXT:   %1 = getelementptr inbounds ptr, ptr %malloccall_fromtape, i64 %0
;CHECK-NEXT:   %malloccall = load ptr, ptr %1, align 8
;CHECK-NEXT:   %i10 = trunc i64 %arg2 to i32
;CHECK-NEXT:   %i11 = icmp sgt i32 %i10, 0
;CHECK-NEXT:   br i1 %i11, label %bb12, label %invertbb73

;CHECK: bb12:                                             ; preds = %bb
;CHECK-NEXT:   %i13 = add nsw i32 %i10, -1
;CHECK-NEXT:   store i32 0, ptr %i7, align 4
;CHECK-NEXT:   %i14 = load i32, ptr %arg, align 4
;CHECK-NEXT:   store i32 0, ptr %i_smpl, align 4
;CHECK-NEXT:   store i32 %i13, ptr %i5_smpl, align 4
;CHECK-NEXT:   store i32 1, ptr %i6_smpl, align 4
;CHECK-NEXT:   call void @__kmpc_for_static_init_4(ptr nonnull @1, i32 %i14, i32 34, ptr nonnull %i7, ptr nocapture nonnull %i_smpl, ptr nocapture nonnull %i5_smpl, ptr nocapture nonnull %i6_smpl, i32 1, i32 1) #14, !noalias !14
;CHECK-NEXT:   %2 = load i32, ptr %i5_smpl, align 4, !invariant.group !61
;CHECK-NEXT:   %3 = load i32, ptr %i_smpl, align 4, !invariant.group !62
;CHECK-NEXT:   %i16 = call i32 @llvm.smin.i32(i32 %2, i32 %i13) #14
;CHECK-NEXT:   %i18 = icmp slt i32 %i16, %3
;CHECK-NEXT:   br i1 %i18, label %bb63, label %bb19

;CHECK: bb19:                                             ; preds = %bb12
;CHECK-NEXT:   %i22 = add i32 %i16, 1
;CHECK-NEXT:   %i24 = sub i32 %i22, %3
;CHECK-NEXT:   %i25 = sub i32 %i16, %3
;CHECK-NEXT:   %i26 = and i32 %i24, 3
;CHECK-NEXT:   %i27 = icmp eq i32 %i26, 0
;CHECK-NEXT:   br i1 %i27, label %bb38, label %bb28

;CHECK: bb28:                                             ; preds = %bb19, %bb28
;CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb28 ], [ 0, %bb19 ]
;CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
;CHECK-NEXT:   %4 = trunc i64 %iv to i32
;CHECK-NEXT:   %i36 = add i32 %4, 1
;CHECK-NEXT:   %i37 = icmp eq i32 %i36, %i26
;CHECK-NEXT:   br i1 %i37, label %bb38, label %bb28, !llvm.loop !19

;CHECK: bb38:                                             ; preds = %bb28, %bb19
;CHECK-NEXT:   %i40_fromtape = extractvalue { ptr, ptr, ptr } %truetape, 2
;CHECK-NEXT:   %5 = getelementptr inbounds i64, ptr %i40_fromtape, i64 %0
;CHECK-NEXT:   %i40 = load i64, ptr %5, align 8, !invariant.group !63
;CHECK-NEXT:   %i41 = icmp ult i32 %i25, 3
;CHECK-NEXT:   br i1 %i41, label %bb63, label %bb42

;CHECK: bb42:                                             ; preds = %bb38, %bb42
;CHECK-NEXT:   %iv2 = phi i64 [ %iv.next3, %bb42 ], [ 0, %bb38 ]
;CHECK-NEXT:   %iv.next3 = add nuw nsw i64 %iv2, 1
;CHECK-NEXT:   %6 = shl i64 %iv2, 2
;CHECK-NEXT:   %7 = add nsw i64 %i40, %6
;CHECK-NEXT:   %i60 = add nsw i64 %7, 4
;CHECK-NEXT:   %i61 = trunc i64 %i60 to i32
;CHECK-NEXT:   %i62 = icmp eq i32 %i22, %i61
;CHECK-NEXT:   br i1 %i62, label %bb63, label %bb42

;CHECK: bb63:                                             ; preds = %bb42, %bb38, %bb12
;CHECK-NEXT:   %_cache.0 = phi i8 [ 0, %bb12 ], [ 1, %bb38 ], [ 2, %bb42 ]
;CHECK-NEXT:   %i64_fromtape = extractvalue { ptr, ptr, ptr } %truetape, 1
;CHECK-NEXT:   %8 = getelementptr inbounds i32, ptr %i64_fromtape, i64 %0
;CHECK-NEXT:   %i64 = load i32, ptr %8, align 4
;CHECK-NEXT:   br label %invertbb73

;CHECK: invertbb:                                         ; preds = %invertbb73, %invertbb12
;CHECK-NEXT:   call void @free(ptr %malloccall)
;CHECK-NEXT:   ret void

;CHECK: invertbb12:                                       ; preds = %invertbb28, %invertbb38, %invertbb63
;CHECK-NEXT:   %i14_unwrap = load i32, ptr %arg, align 4, !tbaa !17, !alias.scope !55, !noalias !58, !invariant.group !60
;CHECK-NEXT:   call void @__kmpc_for_static_fini(ptr @1, i32 %i14_unwrap)
;CHECK-NEXT:   br label %invertbb

;CHECK: invertbb28:                                       ; preds = %invertbb38.loopexit, %incinvertbb28
;CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap4, %invertbb38.loopexit ], [ %10, %incinvertbb28 ]
;CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
;CHECK-NEXT:   br i1 %9, label %invertbb12, label %incinvertbb28
;CHECK-NEXT: 
;CHECK-NEXT: incinvertbb28:                                    ; preds = %invertbb28
;CHECK-NEXT:   %10 = add nsw i64 %"iv'ac.0", -1
;CHECK-NEXT:   br label %invertbb28
;CHECK-NEXT: 
;CHECK-NEXT: invertbb38.loopexit:                              ; preds = %invertbb38
;CHECK-NEXT:   %_unwrap3 = add nsw i32 %i26_unwrap10, -1
;CHECK-NEXT:   %_unwrap4 = zext i32 %_unwrap3 to i64
;CHECK-NEXT:   br label %invertbb28
;CHECK-NEXT: 
;CHECK-NEXT: invertbb38:                                       ; preds = %invertbb42, %invertbb63
;CHECK-NEXT:   %_unwrap5 = load i32, ptr %i5_smpl, align 4, !alias.scope !64, !noalias !67, !invariant.group !61
;CHECK-NEXT:   %i13_unwrap6 = add nsw i32 %i10, -1
;CHECK-NEXT:   %11 = call i32 @llvm.smin.i32(i32 %_unwrap5, i32 %i13_unwrap6) #14
;CHECK-NEXT:   %i23_unwrap7 = add i32 %11, 1
;CHECK-NEXT:   %_unwrap8 = load i32, ptr %i_smpl, align 4, !alias.scope !69, !noalias !72, !invariant.group !62
;CHECK-NEXT:   %i24_unwrap9 = sub i32 %i23_unwrap7, %_unwrap8
;CHECK-NEXT:   %i26_unwrap10 = and i32 %i24_unwrap9, 3
;CHECK-NEXT:   %i27_unwrap = icmp eq i32 %i26_unwrap10, 0
;CHECK-NEXT:   br i1 %i27_unwrap, label %invertbb12, label %invertbb38.loopexit
;CHECK-NEXT: 
;CHECK-NEXT: invertbb42:                                       ; preds = %invertbb63.loopexit, %incinvertbb42
;CHECK-NEXT:   %"iv2'ac.0" = phi i64 [ %_unwrap17, %invertbb63.loopexit ], [ %13, %incinvertbb42 ]
;CHECK-NEXT:   %12 = icmp eq i64 %"iv2'ac.0", 0
;CHECK-NEXT:   br i1 %12, label %invertbb38, label %incinvertbb42
;CHECK-NEXT: 
;CHECK-NEXT: incinvertbb42:                                    ; preds = %invertbb42
;CHECK-NEXT:   %13 = add nsw i64 %"iv2'ac.0", -1
;CHECK-NEXT:   br label %invertbb42
;CHECK-NEXT: 
;CHECK-NEXT: invertbb63.loopexit:                              ; preds = %invertbb63
;CHECK-NEXT:   %_unwrap11 = load i32, ptr %i5_smpl, align 4, !alias.scope !64, !noalias !67, !invariant.group !61
;CHECK-NEXT:   %i13_unwrap12 = add nsw i32 %i10, -1
;CHECK-NEXT:   %14 = call i32 @llvm.smin.i32(i32 %_unwrap11, i32 %i13_unwrap12) #14
;CHECK-NEXT:   %_unwrap13 = add i32 %14, -3
;CHECK-NEXT:   %i40_fromtape_unwrap = extractvalue { ptr, ptr, ptr } %truetape, 2
;CHECK-NEXT:   %_unwrap18 = getelementptr inbounds i64, ptr %i40_fromtape_unwrap, i64 %0
;CHECK-NEXT:   %i40_unwrap = load i64, ptr %_unwrap18, align 8, !alias.scope !74, !noalias !77, !invariant.group !63
;CHECK-NEXT:   %_unwrap14 = trunc i64 %i40_unwrap to i32
;CHECK-NEXT:   %_unwrap15 = sub i32 %_unwrap13, %_unwrap14
;CHECK-NEXT:   %_unwrap16 = zext i32 %_unwrap15 to i64
;CHECK-NEXT:   %_unwrap17 = lshr i64 %_unwrap16, 2
;CHECK-NEXT:   br label %invertbb42
;CHECK-NEXT: 
;CHECK-NEXT: invertbb63:                                       ; preds = %invertbb72, %invertbb65
;CHECK-NEXT:   switch i8 %_cache.1, label %invertbb63.loopexit [
;CHECK-NEXT:     i8 0, label %invertbb12
;CHECK-NEXT:     i8 1, label %invertbb38
;CHECK-NEXT:   ]
;CHECK-NEXT: 
;CHECK-NEXT: invertbb65:                                       ; preds = %invertbb72
;CHECK-NEXT:   %15 = load double, ptr %"arg3'", align 8, !tbaa !12, !alias.scope !79, !noalias !82
;CHECK-NEXT:   store double 0.000000e+00, ptr %"arg3'", align 8, !tbaa !12, !alias.scope !79, !noalias !82
;CHECK-NEXT:   %16 = atomicrmw fadd ptr %"arg3'", double %15 monotonic, align 8
;CHECK-NEXT:   br label %invertbb63
;CHECK-NEXT: 
;CHECK-NEXT: invertbb72:                                       ; preds = %invertbb73
;CHECK-NEXT:   %cond = icmp eq i32 %i64_cache.0, 1
;CHECK-NEXT:   br i1 %cond, label %invertbb65, label %invertbb63
;CHECK-NEXT: 
;CHECK-NEXT: invertbb73:                                       ; preds = %bb, %bb63
;CHECK-NEXT:   %i64_cache.0 = phi i32 [ %i64, %bb63 ], [ undef, %bb ]
;CHECK-NEXT:   %_cache.1 = phi i8 [ %_cache.0, %bb63 ], [ undef, %bb ]
;CHECK-NEXT:   br i1 %i11, label %invertbb72, label %invertbb
;CHECK-NEXT: }
