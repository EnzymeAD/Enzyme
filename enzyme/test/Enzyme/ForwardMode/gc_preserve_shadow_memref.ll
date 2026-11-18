; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s

; Verify that in forward mode @llvm.julia.gc_preserve_begin preserves:
; 1. The shadow of an active GC-managed object (%obj) even when its raw data
;    pointer was extracted before the preserve region.
; 2. The shadow GC owner loaded from an active MemoryRef-like struct
;    ({ ptr, ptr addrspace(10) }) when memmove operates on field 0 (raw data
;    pointer) while field 1 (GC owner) is loaded solely for gc_preserve_begin.

%MemRef = type { ptr, ptr addrspace(10) }

declare token @llvm.julia.gc_preserve_begin(...)
declare void @llvm.julia.gc_preserve_end(token)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1 immarg)
declare void @__enzyme_fwddiff(...)

@enzyme_dup = global i32 0, align 4

define void @copy_memref(ptr nocapture readonly %dst_ref, ptr nocapture readonly %src_ref) {
entry:
  %dst_ptr = load ptr, ptr %dst_ref, align 8
  %dst_mem_slot = getelementptr inbounds %MemRef, ptr %dst_ref, i64 0, i32 1
  %dst_mem = load ptr addrspace(10), ptr %dst_mem_slot, align 8

  %src_ptr = load ptr, ptr %src_ref, align 8
  %src_mem_slot = getelementptr inbounds %MemRef, ptr %src_ref, i64 0, i32 1
  %src_mem = load ptr addrspace(10), ptr %src_mem_slot, align 8

  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_mem, ptr addrspace(10) %src_mem)
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %dst_ptr, ptr align 8 %src_ptr, i64 16, i1 false)
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

define void @preserve_direct_obj(ptr addrspace(10) %dst_obj, double %x) {
entry:
  %raw = addrspacecast ptr addrspace(10) %dst_obj to ptr addrspace(11)
  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_obj)
  store double %x, ptr addrspace(11) %raw, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

define void @caller(ptr %d, ptr %dd, ptr %s, ptr %ds, ptr addrspace(10) %o, ptr addrspace(10) %do, double %x, double %dx) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  call void (...) @__enzyme_fwddiff(ptr @copy_memref, i32 %dup, ptr %d, ptr %dd, i32 %dup, ptr %s, ptr %ds)
  call void (...) @__enzyme_fwddiff(ptr @preserve_direct_obj, i32 %dup, ptr addrspace(10) %o, ptr addrspace(10) %do, i32 %dup, double %x, double %dx)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffecopy_memref(ptr nocapture readonly %dst_ref, ptr nocapture %"dst_ref'", ptr nocapture readonly %src_ref, ptr nocapture %"src_ref'")
; CHECK: %"dst_mem_slot'ipg" = getelementptr inbounds %MemRef, ptr %"dst_ref'", i64 0, i32 1
; CHECK: [[DST_SH:%.+]] = load ptr addrspace(10), ptr %"dst_mem_slot'ipg", align 8
; CHECK: %"src_mem_slot'ipg" = getelementptr inbounds %MemRef, ptr %"src_ref'", i64 0, i32 1
; CHECK: [[SRC_SH:%.+]] = load ptr addrspace(10), ptr %"src_mem_slot'ipg", align 8
; CHECK: [[TOK:%.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_mem, ptr addrspace(10) [[DST_SH]], ptr addrspace(10) %src_mem, ptr addrspace(10) [[SRC_SH]])
; CHECK: call void @llvm.memmove.p0.p0.i64(
; CHECK: call void @llvm.memmove.p0.p0.i64(
; CHECK: call void @llvm.julia.gc_preserve_end(token [[TOK]])

; CHECK-LABEL: define internal void @fwddiffepreserve_direct_obj(ptr addrspace(10) %dst_obj, ptr addrspace(10) %"dst_obj'", double %x, double %"x'")
; CHECK: [[TOK2:%.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_obj, ptr addrspace(10) %"dst_obj'")
; CHECK: store double %"x'", ptr addrspace(11) %"raw'ipc", align 8
; CHECK: call void @llvm.julia.gc_preserve_end(token [[TOK2]])
