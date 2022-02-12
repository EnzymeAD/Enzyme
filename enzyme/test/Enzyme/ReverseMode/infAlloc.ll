; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -loop-deletion -simplifycfg -instsimplify -adce -S | FileCheck %s

source_filename = "mem.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local double @infLoop(double noundef %rho0, i64 noundef %numReg) #0 {
entry:
  %cmp3 = icmp ult i64 0, %numReg
  br i1 %cmp3, label %for.body.lr.ph, label %for.end8

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.end
  %r.04 = phi i64 [ 0, %for.body.lr.ph ], [ %inc7, %for.end ]
  %call = call noalias align 16 i8* @calloc(i64 noundef 8, i64 noundef 1000000) #3
  %i4 = bitcast i8* %call to double*
  store double 1.000000e+00, double* %i4, align 8
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %i.01 = phi i64 [ 1, %for.body ], [ %inc, %for.body3 ]
  %sub = sub i64 %i.01, 1
  %arrayidx4 = getelementptr inbounds double, double* %i4, i64 %sub
  %i10 = load double, double* %arrayidx4, align 8
  %mul = fmul double %i10, %rho0
  %arrayidx5 = getelementptr inbounds double, double* %i4, i64 %i.01
  store double %mul, double* %arrayidx5, align 8
  %inc = add i64 %i.01, 1
  %cmp2 = icmp ult i64 %inc, 1000000
  br i1 %cmp2, label %for.body3, label %for.end, !llvm.loop !4

for.end:                                          ; preds = %for.body3
  call void @free(i8* noundef %call) #3
  %inc7 = add i64 %r.04, 1
  %cmp = icmp ult i64 %inc7, %numReg
  br i1 %cmp, label %for.body, label %for.cond.for.end8_crit_edge, !llvm.loop !6

for.cond.for.end8_crit_edge:                      ; preds = %for.end
  br label %for.end8

for.end8:                                         ; preds = %for.cond.for.end8_crit_edge, %entry
  ret double %rho0
}

; Function Attrs: nounwind
declare dso_local noalias i8* @calloc(i64 noundef, i64 noundef) #1

; Function Attrs: nounwind
declare dso_local void @free(i8* noundef) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call double @__enzyme_autodiff(i8* noundef bitcast (double (double, i64)* @infLoop to i8*), double noundef 2.000000e+00, i64 noundef 10000000)
  %call1 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double noundef %call)
  ret i32 0
}

declare dso_local i32 @printf(i8* noundef, ...) #2

declare dso_local double @__enzyme_autodiff(i8* noundef, double noundef, i64 noundef) #2

attributes #0 = { nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.0 (git@github.com:jdoerfert/llvm-project b5b6dc5cda07dc505cc24f6960980780f3d58f3a)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
