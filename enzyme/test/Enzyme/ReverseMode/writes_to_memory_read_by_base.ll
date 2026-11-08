; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

define void @compute(ptr noalias nocapture readonly %params, ptr noalias nocapture %out, i64 %off, i64 %n) {
entry:
  ; Chain > 6 GEPs on %params to exceed BasicAA's MaxLookup depth.
  %p1 = getelementptr inbounds float, ptr %params, i64 %off
  %p2 = getelementptr inbounds float, ptr %p1, i64 1
  %p3 = getelementptr inbounds float, ptr %p2, i64 1
  %p4 = getelementptr inbounds float, ptr %p3, i64 1
  %p5 = getelementptr inbounds float, ptr %p4, i64 1
  %p6 = getelementptr inbounds float, ptr %p5, i64 1
  %p7 = getelementptr inbounds float, ptr %p6, i64 1
  %p8 = getelementptr inbounds float, ptr %p7, i64 1
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %w_gep = getelementptr inbounds float, ptr %p8, i64 %i
  %w = load float, ptr %w_gep, align 4
  %sq = fmul float %w, %w
  %out_gep = getelementptr inbounds float, ptr %out, i64 %i
  store float %sq, ptr %out_gep, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @diff(ptr %params, ptr %dparams, ptr %out, ptr %dout, i64 %off, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @compute, metadata !"enzyme_dup", ptr %params, ptr %dparams, metadata !"enzyme_dup", ptr %out, ptr %dout, i64 %off, i64 %n)
  ret void
}

; CHECK-LABEL: define internal void @diffecompute(
; CHECK-NOT: call noalias nonnull ptr @malloc
; CHECK: ret void
