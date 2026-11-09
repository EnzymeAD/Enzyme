; RUN: if [ %llvmver -ge 22 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global float undef, align 4

declare i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32, i1)
declare i1 @llvm.nvvm.barrier.cta.red.and.aligned.count(i32, i32, i1)
declare i1 @llvm.nvvm.barrier.cta.red.or.aligned.count(i32, i32, i1)
declare float @__enzyme_autodiff(ptr, ...)

define float @cta_red_barriers(float %x, ptr %bar_cfg, ptr %cnt_cfg, i1 %pred) {
entry:
  store float %x, ptr addrspace(3) @shared, align 4
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop ]
  %bslot = getelementptr inbounds i32, ptr %bar_cfg, i64 %iv
  %bar = load i32, ptr %bslot, align 4
  store i32 0, ptr %bslot, align 4
  %cslot = getelementptr inbounds i32, ptr %cnt_cfg, i64 %iv
  %cnt = load i32, ptr %cslot, align 4
  store i32 0, ptr %cslot, align 4
  %votes = call i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32 %bar, i1 %pred)
  %all = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.count(i32 %bar, i32 %cnt, i1 %pred)
  %any = call i1 @llvm.nvvm.barrier.cta.red.or.aligned.count(i32 %bar, i32 %cnt, i1 %all)
  %cur = load float, ptr addrspace(3) @shared, align 4
  %dbl = fadd float %cur, %cur
  store float %dbl, ptr addrspace(3) @shared, align 4
  %iv_next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv_next, 2
  br i1 %done, label %exit, label %loop

exit:
  %val = load float, ptr addrspace(3) @shared, align 4
  ret float %val
}

define float @test(float %x, ptr %bar_cfg, ptr %cnt_cfg, i1 %pred) {
  %dx = call float (ptr, ...) @__enzyme_autodiff(ptr @cta_red_barriers, float %x, metadata !"enzyme_const", ptr %bar_cfg, metadata !"enzyme_const", ptr %cnt_cfg, i1 %pred)
  ret float %dx
}

; CHECK-LABEL: define internal { float } @diffecta_red_barriers(
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; CHECK: invertloop:
; CHECK: %[[REV_BAR:.+]] = load i32, ptr %{{.+}}, align 4
; CHECK: %[[REV_CNT:.+]] = load i32, ptr %{{.+}}, align 4
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %[[REV_BAR]], i32 %[[REV_CNT]])
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %[[REV_BAR]], i32 %[[REV_CNT]])
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %[[REV_BAR]])
; CHECK: ret { float }
