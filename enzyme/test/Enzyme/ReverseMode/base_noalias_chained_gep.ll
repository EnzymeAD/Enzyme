; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

define internal fastcc void @matmul_step(ptr noalias nocapture writeonly %out, ptr noalias nocapture readonly %weight, i64 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %w_gep = getelementptr inbounds float, ptr %weight, i64 %i
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

define internal fastcc void @residual_step(ptr noalias nocapture %out, i64 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %out_gep = getelementptr inbounds float, ptr %out, i64 %i
  %v = load float, ptr %out_gep, align 4
  %v2 = fadd float %v, 1.000000e+00
  store float %v2, ptr %out_gep, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @caller(ptr noalias nocapture %params, ptr noalias nocapture %acts, float %init, i64 %off, i64 %n) {
entry:
  %v = fadd float %init, 1.000000e+00
  store float %v, ptr %params, align 4
  %p1 = getelementptr inbounds float, ptr %params, i64 %off
  store float %v, ptr %p1, align 4
  %p2 = getelementptr inbounds float, ptr %p1, i64 %off
  store float %v, ptr %p2, align 4
  %p3 = getelementptr inbounds float, ptr %p2, i64 %off
  store float %v, ptr %p3, align 4
  %p4 = getelementptr inbounds float, ptr %p3, i64 %off
  store float %v, ptr %p4, align 4
  %p5 = getelementptr inbounds float, ptr %p4, i64 %off
  store float %v, ptr %p5, align 4
  %p6 = getelementptr inbounds float, ptr %p5, i64 %off
  store float %v, ptr %p6, align 4
  %p7 = getelementptr inbounds float, ptr %p6, i64 %off
  store float %v, ptr %p7, align 4
  %p8 = getelementptr inbounds float, ptr %p7, i64 %off
  store float %v, ptr %p8, align 4
  %a1 = getelementptr inbounds float, ptr %acts, i64 %off
  store float %v, ptr %a1, align 4
  %a2 = getelementptr inbounds float, ptr %a1, i64 %off
  store float %v, ptr %a2, align 4
  %a3 = getelementptr inbounds float, ptr %a2, i64 %off
  store float %v, ptr %a3, align 4
  %a4 = getelementptr inbounds float, ptr %a3, i64 %off
  store float %v, ptr %a4, align 4
  %a5 = getelementptr inbounds float, ptr %a4, i64 %off
  store float %v, ptr %a5, align 4
  %a6 = getelementptr inbounds float, ptr %a5, i64 %off
  store float %v, ptr %a6, align 4
  %a7 = getelementptr inbounds float, ptr %a6, i64 %off
  store float %v, ptr %a7, align 4
  %a8 = getelementptr inbounds float, ptr %a7, i64 %off
  store float %v, ptr %a8, align 4
  %a9 = getelementptr inbounds float, ptr %a8, i64 %off
  store float %v, ptr %a9, align 4
  %a10 = getelementptr inbounds float, ptr %a9, i64 %off
  store float %v, ptr %a10, align 4
  %a11 = getelementptr inbounds float, ptr %a10, i64 %off
  store float %v, ptr %a11, align 4
  %a12 = getelementptr inbounds float, ptr %a11, i64 %off
  store float %v, ptr %a12, align 4
  %a13 = getelementptr inbounds float, ptr %a12, i64 %off
  store float %v, ptr %a13, align 4
  %a14 = getelementptr inbounds float, ptr %a13, i64 %off
  store float %v, ptr %a14, align 4
  call fastcc void @matmul_step(ptr %a14, ptr %p8, i64 %n)
  call fastcc void @residual_step(ptr %a14, i64 %n)
  ret void
}

define void @load_caller(ptr noalias nocapture %params, ptr noalias nocapture %acts, float %init, i64 %off, i64 %n) {
entry:
  %v = fadd float %init, 1.000000e+00
  store float %v, ptr %params, align 4
  %a1 = getelementptr inbounds float, ptr %acts, i64 %off
  store float %v, ptr %a1, align 4
  %a2 = getelementptr inbounds float, ptr %a1, i64 %off
  store float %v, ptr %a2, align 4
  %a3 = getelementptr inbounds float, ptr %a2, i64 %off
  store float %v, ptr %a3, align 4
  %a4 = getelementptr inbounds float, ptr %a3, i64 %off
  store float %v, ptr %a4, align 4
  %a5 = getelementptr inbounds float, ptr %a4, i64 %off
  store float %v, ptr %a5, align 4
  %a6 = getelementptr inbounds float, ptr %a5, i64 %off
  store float %v, ptr %a6, align 4
  %a7 = getelementptr inbounds float, ptr %a6, i64 %off
  store float %v, ptr %a7, align 4
  %a8 = getelementptr inbounds float, ptr %a7, i64 %off
  store float %v, ptr %a8, align 4
  %a9 = getelementptr inbounds float, ptr %a8, i64 %off
  store float %v, ptr %a9, align 4
  %a10 = getelementptr inbounds float, ptr %a9, i64 %off
  store float %v, ptr %a10, align 4
  %a11 = getelementptr inbounds float, ptr %a10, i64 %off
  store float %v, ptr %a11, align 4
  %a12 = getelementptr inbounds float, ptr %a11, i64 %off
  store float %v, ptr %a12, align 4
  %a13 = getelementptr inbounds float, ptr %a12, i64 %off
  store float %v, ptr %a13, align 4
  %a14 = getelementptr inbounds float, ptr %a13, i64 %off
  store float %v, ptr %a14, align 4
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p_gep = getelementptr inbounds float, ptr %params, i64 %i
  %w = load float, ptr %p_gep, align 4
  %sq = fmul float %w, %w
  %o_gep = getelementptr inbounds float, ptr %a14, i64 %i
  store float %sq, ptr %o_gep, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  call fastcc void @residual_step(ptr %a14, i64 %n)
  ret void
}

define void @diff(ptr %params, ptr %dparams, ptr %acts, ptr %dacts, float %init, i64 %off, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @caller, metadata !"enzyme_dup", ptr %params, ptr %dparams, metadata !"enzyme_dup", ptr %acts, ptr %dacts, metadata !"enzyme_const", float %init, i64 %off, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @load_caller, metadata !"enzyme_dup", ptr %params, ptr %dparams, metadata !"enzyme_dup", ptr %acts, ptr %dacts, metadata !"enzyme_const", float %init, i64 %off, i64 %n)
  ret void
}

attributes #0 = { argmemonly nounwind }

; CHECK-NOT: @malloc
; CHECK-LABEL: define internal fastcc void @diffematmul_step(
; CHECK-NOT: @malloc
; CHECK: ret void
; CHECK-LABEL: define internal void @diffeload_caller(
; CHECK-NOT: @malloc
; CHECK: ret void
