; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -opaque-pointers -S | FileCheck %s; fi

%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor13 = type { ptr, ptr, [14 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor12 = type { ptr, ptr, [13 x i8] }
%struct.Object = type { ptr, double }

$"?eval@Object1@@UEAANN@Z" = comdat any

$"??_7Object1@@6B@" = comdat largest

$"??_R4Object1@@6B@" = comdat any

$"??_R0?AUObject1@@@8" = comdat any

$"??_R3Object1@@8" = comdat any

$"??_R2Object1@@8" = comdat any

$"??_R1A@?0A@EA@Object1@@8" = comdat any

$"??_R1A@?0A@EA@Object@@8" = comdat any

$"??_R0?AUObject@@@8" = comdat any

$"??_R3Object@@8" = comdat any

$"??_R2Object@@8" = comdat any

$"??_7Object@@6B@" = comdat largest

$"??_R4Object@@6B@" = comdat any

@0 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_7Object1@@6B@", ptr @"?eval@Object1@@UEAANN@Z"] }, comdat($"??_7Object1@@6B@")
@"??_R4Object1@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUObject1@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3Object1@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4Object1@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_7type_info@@6B@" = external constant ptr
@"??_R0?AUObject1@@@8" = linkonce_odr global %rtti.TypeDescriptor13 { ptr @"??_7type_info@@6B@", ptr null, [14 x i8] c".?AUObject1@@\00" }, comdat
@__ImageBase = external dso_local constant i8
@"??_R3Object1@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2Object1@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R2Object1@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@Object1@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@Object@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"??_R1A@?0A@EA@Object1@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUObject1@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3Object1@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R1A@?0A@EA@Object@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUObject@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3Object@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"??_R0?AUObject@@@8" = linkonce_odr constant %rtti.TypeDescriptor12 { ptr @"??_7type_info@@6B@", ptr null, [13 x i8] c".?AUObject@@\00" }, comdat
@"??_R3Object@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R2Object@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)}, comdat
@"??_R2Object@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R1A@?0A@EA@Object@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4Object@@6B@", ptr @_purecall] }, comdat($"??_7Object@@6B@")
@"??_R4Object@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUObject@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R3Object@@8" to i64), i64 ptrtoint( ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R4Object@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat

@"??_7Object1@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
@"??_7Object@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @1, i32 0, i32 0, i32 1)

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef double @"?new_delete_test@@YANPEAN_K@Z"(ptr nocapture noundef readonly %x, i64 noundef %n) #0 {
entry:
  %0 = tail call {i64, i1} @llvm.umul.with.overflow.i64(i64 %n, i64 8)
  %1 = extractvalue { i64, i1 } %0, 1
  %2 = extractvalue { i64, i1 } %0, 0
  %3 = select i1 %1, i64 -1, i64 %2
  %call = tail call noalias noundef nonnull ptr @"??_U@YAPEAX_K@Z"(i64 noundef %3) #6
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.body ], [ 0, %entry ]
  %call1 = tail call noalias noundef nonnull dereferenceable(8) ptr @"??2@YAPEAX_K@Z"(i64 noundef 8) #6
  %arrayidx = getelementptr inbounds ptr, ptr %call, i64 %indvars.iv
  store ptr %call1, ptr %arrayidx, align 8, !tbaa !5
  %arrayidx3 = getelementptr inbounds double, ptr %x, i64 %indvars.iv
  %4 = load double, ptr %arrayidx3, align 8, !tbaa !9
  %mul = fmul double %4, %4
  store double %mul, ptr %arrayidx3, align 8, !tbaa !9
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %cmp = icmp ult i64 %indvars.iv.next, %n
  br i1 %cmp, label %do.body, label %do.body8.preheader, !llvm.loop !11

do.body8.preheader:                               ; preds = %do.body
  %sext = shl i64 %indvars.iv.next, 32
  %5 = ashr exact i64 %sext, 32
  br label %do.body8

do.body8:                                         ; preds = %do.body8.preheader, %do.body8
  %indvars.iv38 = phi i64 [ %5, %do.body8.preheader ], [ %indvars.iv.next39, %do.body8 ]
  %res.0 = phi double [ 0.000000e+00, %do.body8.preheader ], [ %add, %do.body8 ]
  %arrayidx10 = getelementptr inbounds ptr, ptr %call, i64 %indvars.iv38
  %6 = load ptr, ptr %arrayidx10, align 8, !tbaa !5
  %7 = load double, ptr %6, align 8, !tbaa !9
  %add = fadd double %res.0, %7
  tail call void @"??3@YAXPEAX@Z"(ptr noundef %6) #7
  %indvars.iv.next39 = add i64 %indvars.iv38, 1
  %cmp16 = icmp ult i64 %indvars.iv.next39, %n
  br i1 %cmp16, label %do.body8, label %delete.notnull19, !llvm.loop !14

delete.notnull19:
  tail call void @"??_V@YAXPEAX@Z"(ptr noundef nonnull %call) #7
  ret double %add
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #1

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??_U@YAPEAX_K@Z"(i64 noundef) local_unnamed_addr #2

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef) local_unnamed_addr #2

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??3@YAXPEAX@Z"(ptr noundef ) local_unnamed_addr #3

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??_V@YAXPEAX@Z"(ptr noundef ) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @"?new_delete_test_b@@YEAXPEAN0_K@Z"(ptr noundef %x, ptr noundef %x_b, i64 noundef %n) local_unnamed_addr #0 {
entry:
  tail call void @"??$__enzyme_autodiff@XPEANPEAN_K@@YAXPEAXPEAN1_K@Z"(ptr noundef nonnull@"?new_delete_test@@YANPEAN_K@Z", ptr noundef %x, ptr noundef %x_b, i64 noundef %n) #8
  ret void
}

declare dso_local void @"??$__enzyme_autodiff@XPEANPEAN_K@@YAXPEAXPEAN1_K@Z"(ptr noundef, ptr noundef, ptr noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef double @"?eval@@YANAEAUObject@@N@Z"(ptr noundef nonnull align 8 dereferenceable(16) %o, double noundef %v) local_unnamed_addr #0 {
entry:
  %vtable = load ptr, ptr %o, align 8, !tbaa !15
  %0 = load ptr, ptr %vtable, align 8
  %call = tail call noundef double %0(ptr noundef nonnull align 8 dereferenceable(16) %o, double noundef %v) #8
  ret double %call
}

; Function Attrs: nounwind uwtable
define dso_local void @"?aliased_rtti_test@@YAXN@Z"(double noundef %v) local_unnamed_addr #5 {
entry:
  %call4 = tail call noundef ptr @"??$__enzyme_virtualreverse@PEAX@@YAPEAXPEAX@Z"(ptr noundef nonnull @"??_7Object1@@6B@") #8
  ret void
}

declare dso_local noundef ptr @"??$__enzyme_virtualreverse@PEAX@@YAPEAXPEAX@Z"(ptr noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef double @"?eval@Object1@@UEAANN@Z"(ptr noundef nonnull align 8 dereferenceable(16) %this, double noundef %v) unnamed_addr #0 comdat align 2 {
entry:
  %val = getelementptr inbounds %struct.Object, ptr %this, i64 0, i32 1
  %0 = load double, ptr %val, align 8, !tbaa !17
  %mul = fmul double %0, %v
  ret double %mul
}

declare dso_local void @_purecall() unnamed_addr

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { builtin nounwind allocsize(0) }
attributes #7 = { builtin nounwind }
attributes #8 = { nounwind }


!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !7, i64 0}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !12, !13}
!15 = !{!16, !16, i64 0}
!16 = !{!"vtable pointer", !8, i64 0}
!17 = !{!18, !10, i64 8}
!18 = !{!"?AUObject@@", !10, i64 8}

; CHECK: %rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
; CHECK: define internal void @"diffe?new_delete_test@@YANPEAN_K@Z"
; CHECK: define internal { double } @"diffe?eval@Object1@@UEAANN@Z"
