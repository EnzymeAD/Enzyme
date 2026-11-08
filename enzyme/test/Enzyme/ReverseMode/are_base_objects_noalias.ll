; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

define void @compute(ptr noalias nocapture %a, ptr noalias nocapture %b, i64 %off, i64 %n) {
entry:
  ; Chain > 6 GEPs on both %a and %b (without readonly on %a) to exceed BasicAA's MaxLookup depth.
  %a1 = getelementptr inbounds float, ptr %a, i64 %off
  %a2 = getelementptr inbounds float, ptr %a1, i64 1
  %a3 = getelementptr inbounds float, ptr %a2, i64 1
  %a4 = getelementptr inbounds float, ptr %a3, i64 1
  %a5 = getelementptr inbounds float, ptr %a4, i64 1
  %a6 = getelementptr inbounds float, ptr %a5, i64 1
  %a7 = getelementptr inbounds float, ptr %a6, i64 1
  %a8 = getelementptr inbounds float, ptr %a7, i64 1

  %b1 = getelementptr inbounds float, ptr %b, i64 %off
  %b2 = getelementptr inbounds float, ptr %b1, i64 1
  %b3 = getelementptr inbounds float, ptr %b2, i64 1
  %b4 = getelementptr inbounds float, ptr %b3, i64 1
  %b5 = getelementptr inbounds float, ptr %b4, i64 1
  %b6 = getelementptr inbounds float, ptr %b5, i64 1
  %b7 = getelementptr inbounds float, ptr %b6, i64 1
  %b8 = getelementptr inbounds float, ptr %b7, i64 1
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %a_gep = getelementptr inbounds float, ptr %a8, i64 %i
  %v = load float, ptr %a_gep, align 4
  %sq = fmul float %v, %v
  %b_gep = getelementptr inbounds float, ptr %b8, i64 %i
  store float %sq, ptr %b_gep, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @diff(ptr %a, ptr %da, ptr %b, ptr %db, i64 %off, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @compute, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %b, ptr %db, i64 %off, i64 %n)
  ret void
}

; CHECK-LABEL: define internal void @diffecompute(
; CHECK-NOT: call noalias nonnull ptr @malloc
; CHECK: ret void
