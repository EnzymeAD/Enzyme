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
