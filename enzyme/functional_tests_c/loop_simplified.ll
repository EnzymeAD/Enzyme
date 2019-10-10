; ModuleID = 'build/insertsort_sum_alt-enzyme0.ll'
source_filename = "insertsort_sum_alt.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [21 x i8] c"The total sum is %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [26 x i8] c"Diffe for index %d is %f\0A\00", align 1
@.str.2 = private unnamed_addr constant [18 x i8] c"d_array[i] == 0.0\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"insertsort_sum_alt.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.4 = private unnamed_addr constant [18 x i8] c"d_array[i] == 1.0\00", align 1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @insertion_sort_inner(float* nocapture %array, i32 %i) local_unnamed_addr #2 {
entry:
  %cmp29 = icmp sgt i32 %i, 0
  br i1 %cmp29, label %land.rhs.preheader, label %while.end

land.rhs.preheader:                               ; preds = %entry
  %0 = sext i32 %i to i64
  br label %land.rhs

land.rhs:                                         ; preds = %while.body, %land.rhs.preheader
  %indvars.iv = phi i64 [ %0, %land.rhs.preheader ], [ %indvars.iv.next, %while.body ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds float, float* %array, i64 %indvars.iv.next
  %1 = load float, float* %arrayidx, align 4, !tbaa !2
  %arrayidx2 = getelementptr inbounds float, float* %array, i64 %indvars.iv
  %2 = load float, float* %arrayidx2, align 4, !tbaa !2
  %cmp3 = fcmp ogt float %1, %2
  br i1 %cmp3, label %while.body, label %while.end.loopexit

while.body:                                       ; preds = %land.rhs
  store float %1, float* %arrayidx2, align 4, !tbaa !2
  store float %2, float* %arrayidx, align 4, !tbaa !2
  %cmp = icmp sgt i64 %indvars.iv, 1
  br i1 %cmp, label %land.rhs, label %while.end.loopexit

while.end.loopexit:                               ; preds = %land.rhs, %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #1

declare double @__enzyme_autodiff.f64(...) local_unnamed_addr

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #0

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
