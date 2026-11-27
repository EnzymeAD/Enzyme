; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1"
target triple = "amdgcn-amd-amdhsa"

@sh_tile = internal addrspace(3) global [64 x float] undef, align 4

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare void @__enzyme_autodiff(ptr, ...)

define void @kernel(ptr %in, ptr %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %tid to i64
  %p_in = getelementptr inbounds float, ptr %in, i64 %idx
  %val = load float, ptr %p_in, align 4
  %slot = getelementptr inbounds [64 x float], ptr addrspace(3) @sh_tile, i64 0, i64 %idx
  store float %val, ptr addrspace(3) %slot, align 4
  call void @llvm.amdgcn.s.barrier()
  %loaded = load float, ptr addrspace(3) %slot, align 4
  %sq = fmul float %loaded, %loaded
  %p_out = getelementptr inbounds float, ptr %out, i64 %idx
  store float %sq, ptr %p_out, align 4
  ret void
}

define amdgpu_kernel void @caller(ptr %in, ptr %din, ptr %out, ptr %dout) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @kernel, ptr %in, ptr %din, ptr %out, ptr %dout)
  ret void
}

; CHECK-LABEL: define internal void @diffekernel(
; CHECK-NEXT: entry:
; CHECK:        %[[TX:.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT:   %[[TY:.+]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT:   %[[TZ:.+]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT:   %[[DP:.+]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:   %[[PX:.+]] = getelementptr inbounds i8, ptr addrspace(4) %[[DP]], i64 4
; CHECK-NEXT:   %[[BX16:.+]] = load i16, ptr addrspace(4) %[[PX]], align 4
; CHECK-NEXT:   %[[BX:.+]] = zext i16 %[[BX16]] to i32
; CHECK-NEXT:   %[[PY:.+]] = getelementptr inbounds i8, ptr addrspace(4) %[[DP]], i64 6
; CHECK-NEXT:   %[[BY16:.+]] = load i16, ptr addrspace(4) %[[PY]], align 2
; CHECK-NEXT:   %[[BY:.+]] = zext i16 %[[BY16]] to i32
; CHECK-NEXT:   %[[PZ:.+]] = getelementptr inbounds i8, ptr addrspace(4) %[[DP]], i64 8
; CHECK-NEXT:   %[[BZ16:.+]] = load i16, ptr addrspace(4) %[[PZ]], align 4
; CHECK-NEXT:   %[[BZ:.+]] = zext i16 %[[BZ16]] to i32
; CHECK-NEXT:   %[[BYTZ:.+]] = mul i32 %[[BY]], %[[TZ]]
; CHECK-NEXT:   %[[TYZ:.+]] = add i32 %[[TY]], %[[BYTZ]]
; CHECK-NEXT:   %[[BXTYZ:.+]] = mul i32 %[[BX]], %[[TYZ]]
; CHECK-NEXT:   %[[TID:.+]] = add i32 %[[TX]], %[[BXTYZ]]
; CHECK-NEXT:   %[[BXBY:.+]] = mul i32 %[[BX]], %[[BY]]
; CHECK-NEXT:   %[[NTHREADS:.+]] = mul i32 %[[BXBY]], %[[BZ]]
; CHECK-NEXT:   %[[C0:.+]] = icmp ult i32 %[[TID]], 64
; CHECK-NEXT:   br i1 %[[C0]], label %shzero.loop, label %[[AFTER:.+]]

; CHECK: shzero.loop:
; CHECK-NEXT:   %[[IDX:.+]] = phi i32 [ %[[TID]], %entry ], [ %[[NEXT:.+]], %shzero.loop ]
; CHECK-NEXT:   %[[PTR:.+]] = getelementptr inbounds i32, ptr addrspace(3) @sh_tile_shadow, i32 %[[IDX]]
; CHECK-NEXT:   store i32 0, ptr addrspace(3) %[[PTR]], align 4
; CHECK-NEXT:   %[[NEXT]] = add i32 %[[IDX]], %[[NTHREADS]]
; CHECK-NEXT:   %[[COND:.+]] = icmp ult i32 %[[NEXT]], 64
; CHECK-NEXT:   br i1 %[[COND]], label %shzero.loop, label %[[AFTER]]

; CHECK: [[AFTER]]:
; CHECK:        call void @llvm.amdgcn.s.barrier()
