; RUN: if [ %llvmver -gt 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi

; LLVM 21 CTA barriers take an explicit barrier ID and, for the counted
; form, a thread count. Preserve these operands in the reverse pass and use
; barrier zero for shared-shadow initialization and reverse reductions.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global float undef, align 4

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)
declare void @llvm.nvvm.barrier.cta.sync.aligned.count(i32, i32)
declare i32 @llvm.nvvm.barrier0.popc(i32)
declare float @__enzyme_autodiff(ptr, ...)

define float @barriers(float %x, i32 %barrier, i32 %count) {
entry:
  store float %x, ptr addrspace(3) @shared, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %barrier)
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %barrier, i32 %count)
  %votes = call i32 @llvm.nvvm.barrier0.popc(i32 1)
  %value = load float, ptr addrspace(3) @shared, align 4
  ret float %value
}

define float @test(float %x, i32 %barrier, i32 %count) {
  %dx = call float (ptr, ...) @__enzyme_autodiff(ptr @barriers, float %x, i32 %barrier, i32 %count)
  ret float %dx
}

define float @barriers_loop_args(float %x, ptr %bar_cfg, ptr %cnt_cfg) {
entry:
  store float %x, ptr addrspace(3) @shared, align 4
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop ]
  %bslot = getelementptr inbounds i32, ptr %bar_cfg, i64 %iv
  %barrier = load i32, ptr %bslot, align 4
  store i32 0, ptr %bslot, align 4
  %cslot = getelementptr inbounds i32, ptr %cnt_cfg, i64 %iv
  %count = load i32, ptr %cslot, align 4
  store i32 0, ptr %cslot, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %barrier)
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %barrier, i32 %count)
  %cur = load float, ptr addrspace(3) @shared, align 4
  %dbl = fadd float %cur, %cur
  store float %dbl, ptr addrspace(3) @shared, align 4
  %iv_next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv_next, 2
  br i1 %done, label %exit, label %loop

exit:
  %value = load float, ptr addrspace(3) @shared, align 4
  ret float %value
}

define float @test_loop(float %x, ptr %bar_cfg, ptr %cnt_cfg) {
  %dx = call float (ptr, ...) @__enzyme_autodiff(ptr @barriers_loop_args, float %x, metadata !"enzyme_const", ptr %bar_cfg, metadata !"enzyme_const", ptr %cnt_cfg)
  ret float %dx
}

; CHECK-LABEL: define internal { float } @diffebarriers(
; CHECK: store float 0.000000e+00, ptr addrspace(3) @shared_shadow
; CHECK: br label %[[START:[a-zA-Z0-9_.]+]]
; CHECK: [[START]]:
; The initialization barrier must use barrier ID zero.
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; CHECK: store float %x, ptr addrspace(3) @shared
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %barrier)
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %barrier, i32 %count)
; LLVM 22 upgrades barrier0.popc to barrier.cta.red.popc.aligned.all.
; CHECK: call i32 @llvm.nvvm.{{.*}}popc{{.*}}(
; The reduction becomes a barrier in reverse, followed by the original
; counted and uncounted barriers in reverse order with their operands intact.
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %barrier, i32 %count)
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %barrier)
; CHECK: ret { float }

; CHECK-LABEL: define internal { float } @diffebarriers_loop_args(
; CHECK: invertloop:
; CHECK: %[[REV_BAR:.+]] = load i32, ptr %{{.+}}, align 4
; CHECK: %[[REV_CNT:.+]] = load i32, ptr %{{.+}}, align 4
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 %[[REV_BAR]], i32 %[[REV_CNT]])
; CHECK-NEXT: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %[[REV_BAR]])
