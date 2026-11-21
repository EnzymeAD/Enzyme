; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; The derivative of an active alpha of a cuBLAS v2 gemm is the inner product of the shadow
; of C with A*B. It is computed with the cuBLAS dot, which takes the handle first and
; writes its result through a trailing pointer.

declare i32 @cublasDgemm_v2(ptr, i32, i32, i32, i32, i32, ptr, ptr, i32, ptr, i32, ptr, ptr, i32)

define void @f(ptr %handle, ptr noalias %C, ptr noalias %A, ptr noalias %B, ptr noalias %alpha, ptr noalias %beta) {
entry:
  %r = call i32 @cublasDgemm_v2(ptr %handle, i32 0, i32 0, i32 4, i32 4, i32 8, ptr %alpha, ptr %A, i32 4, ptr %B, i32 8, ptr %beta, ptr %C, i32 4)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @active(ptr %handle, ptr %C, ptr %dC, ptr %A, ptr %dA, ptr %B, ptr %dB, ptr %alpha, ptr %dalpha, ptr %beta, ptr %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_const", ptr %handle, metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_dup", ptr %B, ptr %dB, metadata !"enzyme_dup", ptr %alpha, ptr %dalpha, metadata !"enzyme_dup", ptr %beta, ptr %dbeta)
  ret void
}

; alpha: the inner product of the shadow of C with A*B, and beta: with the C that was
; overwritten, both through the cuBLAS dot with the handle of the call.
; CHECK: define internal void @diffef(ptr %handle,
; CHECK: call void @cublasDdot_v2(ptr %handle, i32 16, ptr %"C'", i32 1, ptr %mat_AB, i32 1, ptr %[[res1:.+]])
; CHECK: call void @cublasDdot_v2(ptr %handle, i32 16, ptr %"C'", i32 1, ptr %cache.C, i32 1, ptr %[[res2:.+]])

; CHECK: define internal double @__enzyme_inner_prodcublasD_v2(ptr %handle, i32 %blasm, i32 %blasn, ptr noalias nocapture readonly %A, i32 %lda, ptr noalias readonly %B)
; CHECK: %dot.res = alloca double
; CHECK: fast.path:
; CHECK-NEXT: call void @cublasDdot_v2(ptr %handle, i32 %mat.size, ptr %A, i32 1, ptr %B, i32 1, ptr %dot.res)
; CHECK-NEXT: %dot = load double, ptr %dot.res
; CHECK: for.body:
; CHECK: call void @cublasDdot_v2(ptr %handle, i32 %blasm, ptr %A.i, i32 1, ptr %B.i, i32 1, ptr %dot.res)

; CHECK: declare void @cublasDdot_v2(ptr, i32 "enzyme_inactive", ptr nocapture readonly, i32 "enzyme_inactive", ptr nocapture readonly, i32 "enzyme_inactive", ptr nocapture writeonly)
