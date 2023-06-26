; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -correlated-propagation -adce -instsimplify -early-cse -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s -check-prefixes LLVM13,SHARED ; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,correlated-propagation,adce,instsimplify,early-cse,%simplifycfg,correlated-propagation,adce,jump-threading,instsimplify,early-cse,%simplifycfg)" -S | FileCheck %s -check-prefixes LLVM13,SHARED

; ModuleID = 'orig.ll'
source_filename = "../benchmarks/hand/hand.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @dhand_objective(double* %A, double* %dA) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff(void (double*)* nonnull @hand_objective, double* %A, double* %dA) #3
  ret void
}

declare noalias i64* @getsize() #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind uwtable
define internal void @hand_objective(double* %tmp10) #0 {
entry:
  %ncols12 = call noalias i64* @getsize()
  tail call void @mat_mult(i64* %ncols12, double* %tmp10) #3
  tail call void @noop(double* %tmp10) #3
  ret void
}


define internal void @noop(double* noalias %tmp10) {
entry:
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(...) local_unnamed_addr #2

define internal void @mat_mult(i64* noalias readonly %ncols12, double* noalias %tmp10) {
entry:
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.inc30, %entry
  %indvars.iv67 = phi i64 [ 0, %entry ], [ %indvars.iv.next68, %for.inc30 ]
  %wide.trip.count = load i64, i64* %ncols12, align 4, !tbaa !9
  store i64 3, i64* %ncols12
  br label %for.body15

for.body15:                                       ; preds = %for.body15, %for.body11
  %indvars.iv = phi i64 [ 0, %for.cond8.preheader ], [ %indvars.iv.next, %for.body15 ]
  %arrayidx = getelementptr inbounds double, double* %tmp10, i64 %indvars.iv
  %tmp15 = load double, double* %arrayidx, align 8, !tbaa !10
  %mul20 = fmul fast double %tmp15, %tmp15
  store double %mul20, double* %arrayidx, align 8, !tbaa !10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.inc30, label %for.body15

for.inc30:                                        ; preds = %for.end
  %indvars.iv.next68 = add nuw nsw i64 %indvars.iv67, 1
  %exitcond70 = icmp eq i64 %indvars.iv.next68, 13
  br i1 %exitcond70, label %for.end32, label %for.cond8.preheader

for.end32:                                        ; preds = %for.inc30
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { "enzyme_inactive" }
!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !8, i64 8}
!7 = !{!"_ZTS6Matrix", !3, i64 0, !3, i64 4, !8, i64 8}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!7, !3, i64 4}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !4, i64 0}

; SHARED: define internal void @diffemat_mult(i64* noalias readonly %ncols12, double* noalias %tmp10, double* %"tmp10'", { i64*, double** } %tapeArg)
; SHARED-NEXT: entry:
; SHARED-DAG:   %[[i0:.+]] = extractvalue { i64*, double** } %tapeArg, 0
; SHARED-DAG:   %[[i1:.+]] = extractvalue { i64*, double** } %tapeArg, 1
; SHARED:   br label %for.cond8.preheader

; SHARED: for.cond8.preheader:                              ; preds = %for.inc30, %entry
; SHARED-NEXT:   %iv = phi i64 [ %iv.next, %for.inc30 ], [ 0, %entry ]
; SHARED-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; SHARED-NEXT:   %2 = getelementptr inbounds i64, i64* %[[i0]], i64 %iv
; SHARED-NEXT:   %wide.trip.count = load i64, i64* %2, align 8
; SHARED-NEXT:   br label %for.body15

; SHARED: for.body15:                                       ; preds = %for.body15, %for.cond8.preheader
; SHARED-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body15 ], [ 0, %for.cond8.preheader ]
; SHARED-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; SHARED-NEXT:   %exitcond = icmp eq i64 %iv.next2, %wide.trip.count
; SHARED-NEXT:   br i1 %exitcond, label %for.inc30, label %for.body15

; SHARED: for.inc30:                                        ; preds = %for.body15
; SHARED-NEXT:   %exitcond70 = icmp eq i64 %iv.next, 13
; SHARED-NEXT:   br i1 %exitcond70, label %invertfor.inc30, label %for.cond8.preheader

; SHARED: invertentry:                                      ; preds = %invertfor.cond8.preheader
; SHARED-DAG:   %[[a3:.+]] = bitcast i64* %[[i0]] to i8*
; SHARED-DAG:   tail call void @free(i8* nonnull %[[a3]])
; SHARED-DAG:   %[[a4:.+]] = bitcast double** %[[i1]] to i8*
; SHARED-DAG:   tail call void @free(i8* nonnull %[[a4]])
; SHARED-DAG:   ret void

; SHARED: invertfor.cond8.preheader:                        ; preds = %invertfor.body15
; SHARED-NEXT:   %5 = icmp eq i64 %"iv'ac.0", 0
; LLVM13-NEXT:   %[[forfree15:.+]] = load double*, double** %9, align 8
; LLVM13-NEXT:   %6 = bitcast double* %[[forfree15]] to i8*
; SHARED-NEXT:   tail call void @free(i8* nonnull %6)
; SHARED-NEXT:   br i1 %5, label %invertentry, label %incinvertfor.cond8.preheader

; SHARED: incinvertfor.cond8.preheader:                     ; preds = %invertfor.cond8.preheader
; SHARED-NEXT:   %7 = add nsw i64 %"iv'ac.0", -1
; SHARED-NEXT:   br label %invertfor.inc30

; SHARED: invertfor.body15:                                 ; preds = %invertfor.inc30, %incinvertfor.body15
; SHARED-NEXT:   %"iv1'ac.0" = phi i64 [ %[[unwrap18:.+]], %invertfor.inc30 ], [ %[[i15:.+]], %incinvertfor.body15 ]
; SHARED-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"tmp10'", i64 %"iv1'ac.0"
; SHARED-NEXT:   %8 = load double, double* %"arrayidx'ipg_unwrap", align 8
; SHARED-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; SHARED-NEXT:   %9 = getelementptr inbounds double*, double** %[[i1]], i64 %"iv'ac.0"
; SHARED-NEXT:   %10 = load double*, double** %9, align 8
; SHARED-NEXT:   %11 = getelementptr inbounds double, double* %10, i64 %"iv1'ac.0"
; SHARED-NEXT:   %12 = load double, double* %11, align 8
; SHARED-NEXT:   %[[m0diffetmp15:.+]] = fmul fast double %8, %12
; SHARED-NEXT:   %[[i13:.+]] = fadd fast double %[[m0diffetmp15]], %[[m0diffetmp15]]
; SHARED-NEXT:   store double %[[i13]], double* %"arrayidx'ipg_unwrap", align 8
; SHARED-NEXT:   %[[i14:.+]] = icmp eq i64 %"iv1'ac.0", 0
; SHARED-NEXT:   br i1 %[[i14]], label %invertfor.cond8.preheader, label %incinvertfor.body15

; SHARED: incinvertfor.body15:                              ; preds = %invertfor.body15
; SHARED-NEXT:   %[[i15]] = add nsw i64 %"iv1'ac.0", -1
; SHARED-NEXT:   br label %invertfor.body15

; SHARED: invertfor.inc30:                                  ; preds = %for.inc30, %incinvertfor.cond8.preheader
; SHARED-NEXT:   %"iv'ac.0" = phi i64 [ %7, %incinvertfor.cond8.preheader ], [ 12, %for.inc30 ]
; SHARED-NEXT:   %[[unwrap16:.+]] = getelementptr inbounds i64, i64* %[[i0]], i64 %"iv'ac.0"
; SHARED-NEXT:   %[[unwrap17:.+]] = load i64, i64* %[[unwrap16]], align 8, !tbaa !2, !alias.scope !{{[0-9]+}}, !noalias !{{[0-9]+}}, !invariant.group !
; SHARED-NEXT:   %[[unwrap18]] = add i64 %[[unwrap17]], -1
; SHARED-NEXT:   br label %invertfor.body15
; SHARED-NEXT: }
