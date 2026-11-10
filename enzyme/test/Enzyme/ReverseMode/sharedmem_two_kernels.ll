; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Each kernel's derivative zeroes the shadow of its own shared buffer only, so
; the other kernel's shadow does not count towards its static shared memory.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@tile_a = internal unnamed_addr addrspace(3) global [64 x double] undef, align 8
@tile_b = internal unnamed_addr addrspace(3) global [512 x double] undef, align 8

define void @ka(ptr %x, ptr %y) {
entry:
  %t = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %i = zext i32 %t to i64
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %px, align 8
  %sa = getelementptr inbounds [64 x double], ptr addrspace(3) @tile_a, i64 0, i64 %i
  store double %v, ptr addrspace(3) %sa, align 8
  tail call void @llvm.nvvm.barrier0()
  %w = load double, ptr addrspace(3) %sa, align 8
  %m = fmul double %w, %w
  %py = getelementptr inbounds double, ptr %y, i64 %i
  store double %m, ptr %py, align 8
  ret void
}

define void @kb(ptr %x, ptr %y) {
entry:
  %t = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %i = zext i32 %t to i64
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %px, align 8
  %sb = getelementptr inbounds [512 x double], ptr addrspace(3) @tile_b, i64 0, i64 %i
  store double %v, ptr addrspace(3) %sb, align 8
  tail call void @llvm.nvvm.barrier0()
  %w = load double, ptr addrspace(3) %sb, align 8
  %m = fmul double %w, %v
  %py = getelementptr inbounds double, ptr %y, i64 %i
  store double %m, ptr %py, align 8
  ret void
}

define void @grads(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  tail call void (...) @__enzyme_autodiff(ptr @ka, ptr %x, ptr %dx, ptr %y, ptr %dy)
  tail call void (...) @__enzyme_autodiff(ptr @kb, ptr %x, ptr %dx, ptr %y, ptr %dy)
  ret void
}

declare void @__enzyme_autodiff(...)

declare void @llvm.nvvm.barrier0()

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

; CHECK: @tile_a_shadow = internal unnamed_addr addrspace(3) global [64 x double] undef, align 8
; CHECK: @tile_b_shadow = internal unnamed_addr addrspace(3) global [512 x double] undef, align 8

; CHECK-LABEL: define internal void @diffeka(
; CHECK: shblock:
; CHECK-NEXT: store [64 x double] zeroinitializer, ptr addrspace(3) @tile_a_shadow, align 8
; CHECK-NOT: @tile_b_shadow
; CHECK: ret void

; CHECK-LABEL: define internal void @diffekb(
; CHECK: shblock:
; CHECK-NEXT: store [512 x double] zeroinitializer, ptr addrspace(3) @tile_b_shadow, align 8
; CHECK-NOT: @tile_a_shadow
; CHECK: ret void
