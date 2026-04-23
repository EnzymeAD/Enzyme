; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; the f function is simple enough that no calls to malloc should be required here

; ModuleID = 'nomallocs.c'
source_filename = "nomallocs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_allocated = external local_unnamed_addr global i32, align 4
@enzyme_tape = external local_unnamed_addr global i32, align 4

; Function Attrs: mustprogress nofree nounwind willreturn uwtable
define dso_local void @f(ptr nocapture noundef writeonly %y, ptr nocapture noundef readonly %x) #0 {
entry:
  %0 = load double, ptr %x, align 8, !tbaa !5
  %call = tail call double @sin(double noundef %0) #5
  store double %call, ptr %y, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn writeonly
declare double @sin(double noundef) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local void @f_b(ptr noundef %y, ptr noundef %y_b, ptr noundef %x, ptr noundef %x_b) local_unnamed_addr #2 {
entry:
  %tape = alloca [100 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 100, ptr nonnull %tape) #5
  %0 = load i32, ptr @enzyme_allocated, align 4, !tbaa !9
  %1 = load i32, ptr @enzyme_tape, align 4, !tbaa !9
  %call = call ptr (ptr, ...) @__enzyme_augmentfwd(ptr noundef nonnull @f, ptr noundef %y, ptr noundef %y_b, ptr noundef %x, ptr noundef %x_b, i32 noundef %0, i64 noundef 100, i32 noundef %1, ptr noundef nonnull %tape) #5
  %2 = load i32, ptr @enzyme_allocated, align 4, !tbaa !9
  %3 = load i32, ptr @enzyme_tape, align 4, !tbaa !9
  call void (ptr, ...) @__enzyme_reverse(ptr noundef nonnull @f, ptr noundef %x, ptr noundef %y_b, ptr noundef %x, ptr noundef %x_b, i32 noundef %2, i64 noundef 100, i32 noundef %3, ptr noundef nonnull %tape) #5
  call void @llvm.lifetime.end.p0(i64 100, ptr nonnull %tape) #5
  ret void
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

declare ptr @__enzyme_augmentfwd(ptr noundef, ...) local_unnamed_addr #4

declare void @__enzyme_reverse(ptr noundef, ...) local_unnamed_addr #4

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

attributes #0 = { mustprogress nofree nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 15.0.7"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}

; CHECK-NOT: malloc
