; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tstore_f32 -S -o /dev/null | FileCheck %s --check-prefix=F32
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tstore_f16 -S -o /dev/null | FileCheck %s --check-prefix=F16

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1(ptr addrspace(1), float, float, float, float, float, float, float, float, i32)
declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p1(ptr addrspace(1), <2 x half>, <2 x half>, <2 x half>, <2 x half>, i32)

define void @tstore_f32(ptr addrspace(1) %out, ptr %in, i32 %ld, float %d0, float %d1, float %d2, float %d3, float %d4, float %d5, float %d6, float %d7) {
entry:
  %off = sext i32 %ld to i64
  %p = getelementptr float, ptr %in, i64 %off
  %v = load float, ptr %p, align 4
  call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1(ptr addrspace(1) %out, float %v, float %d1, float %d2, float %d3, float %d4, float %d5, float %d6, float %d7, i32 %ld)
  ret void
}

define void @tstore_f16(ptr addrspace(1) %out, ptr %in, i32 %ld, <2 x half> %d0, <2 x half> %d1, <2 x half> %d2, <2 x half> %d3) {
entry:
  %off = sext i32 %ld to i64
  %p = getelementptr <2 x half>, ptr %in, i64 %off
  %v = load <2 x half>, ptr %p, align 4
  call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p1(ptr addrspace(1) %out, <2 x half> %v, <2 x half> %d1, <2 x half> %d2, <2 x half> %d3, i32 %ld)
  ret void
}

; F32: tstore_f32 - {}
; F32-NEXT: ptr addrspace(1) %out: {[-1]:Pointer, [-1,0]:Float@float}
; F32-NEXT: ptr %in: {[-1]:Pointer}
; F32-NEXT: i32 %ld: {[-1]:Integer}
; F32-NEXT: float %d0: {[-1]:Float@float}
; F32-NEXT: float %d1: {[-1]:Float@float}
; F32-NEXT: float %d2: {[-1]:Float@float}
; F32-NEXT: float %d3: {[-1]:Float@float}
; F32-NEXT: float %d4: {[-1]:Float@float}
; F32-NEXT: float %d5: {[-1]:Float@float}
; F32-NEXT: float %d6: {[-1]:Float@float}
; F32-NEXT: float %d7: {[-1]:Float@float}
; F32-NEXT: entry
; F32-NEXT:   %off = sext i32 %ld to i64: {[-1]:Integer}
; F32-NEXT:   %p = getelementptr float, ptr %in, i64 %off: {[-1]:Pointer, [-1,0]:Float@float}
; F32-NEXT:   %v = load float, ptr %p, align 4: {[-1]:Float@float}
; F32-NEXT:   call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1(ptr addrspace(1) %out, float %v, float %d1, float %d2, float %d3, float %d4, float %d5, float %d6, float %d7, i32 %ld): {}
; F32-NEXT:   ret void: {}

; F16: tstore_f16 - {}
; F16-NEXT: ptr addrspace(1) %out: {[-1]:Pointer, [-1,0]:Float@half}
; F16-NEXT: ptr %in: {[-1]:Pointer}
; F16-NEXT: i32 %ld: {[-1]:Integer}
; F16-NEXT: <2 x half> %d0: {[-1]:Float@half}
; F16-NEXT: <2 x half> %d1: {[-1]:Float@half}
; F16-NEXT: <2 x half> %d2: {[-1]:Float@half}
; F16-NEXT: <2 x half> %d3: {[-1]:Float@half}
; F16-NEXT: entry
; F16-NEXT:   %off = sext i32 %ld to i64: {[-1]:Integer}
; F16-NEXT:   %p = getelementptr <2 x half>, ptr %in, i64 %off: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half}
; F16-NEXT:   %v = load <2 x half>, ptr %p, align 4: {[-1]:Float@half}
; F16-NEXT:   call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p1(ptr addrspace(1) %out, <2 x half> %v, <2 x half> %d1, <2 x half> %d2, <2 x half> %d3, i32 %ld): {}
; F16-NEXT:   ret void: {}
