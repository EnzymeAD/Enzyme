; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s

; Verify that in reverse mode (both augmented forward and reverse passes)
; @llvm.julia.gc_preserve_begin preserves the shadow GC owner loaded from an
; active MemoryRef-like struct ({ ptr, ptr addrspace(10) }).

%MemRef = type { ptr, ptr addrspace(10) }

declare token @llvm.julia.gc_preserve_begin(...)
declare void @llvm.julia.gc_preserve_end(token)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1 immarg)
declare ptr @__enzyme_virtualreverse(...)

define double @copy_memref(ptr nocapture readonly %dst_ref, ptr nocapture readonly %src_ref) {
entry:
  %dst_ptr = load ptr, ptr %dst_ref, align 8
  %dst_mem_slot = getelementptr inbounds %MemRef, ptr %dst_ref, i64 0, i32 1
  %dst_mem = load ptr addrspace(10), ptr %dst_mem_slot, align 8

  %src_ptr = load ptr, ptr %src_ref, align 8
  %src_mem_slot = getelementptr inbounds %MemRef, ptr %src_ref, i64 0, i32 1
  %src_mem = load ptr addrspace(10), ptr %src_mem_slot, align 8

  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_mem, ptr addrspace(10) %src_mem)
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %dst_ptr, ptr align 8 %src_ptr, i64 8, i1 false)
  %res = load double, ptr %dst_ptr, align 8
  %scaled = fmul double %res, 2.000000e+00
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret double %scaled
}

define ptr @test() {
entry:
  %r = call ptr (...) @__enzyme_virtualreverse(ptr @copy_memref)
  ret ptr %r
}

; CHECK-LABEL: define internal {{.+}} @augmented_copy_memref(ptr nocapture readonly %dst_ref, ptr nocapture %"dst_ref'", ptr nocapture readonly %src_ref, ptr nocapture %"src_ref'")
; CHECK: %"dst_mem_slot'ipg" = getelementptr inbounds %MemRef, ptr %"dst_ref'", i64 0, i32 1
; CHECK: [[AUG_DST_SH:%.+]] = load ptr addrspace(10), ptr %"dst_mem_slot'ipg", align 8
; CHECK: %"src_mem_slot'ipg" = getelementptr inbounds %MemRef, ptr %"src_ref'", i64 0, i32 1
; CHECK: [[AUG_SRC_SH:%.+]] = load ptr addrspace(10), ptr %"src_mem_slot'ipg", align 8
; CHECK: [[AUG_TOK:%.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_mem, ptr addrspace(10) [[AUG_DST_SH]], ptr addrspace(10) %src_mem, ptr addrspace(10) [[AUG_SRC_SH]])
; CHECK: call void @llvm.julia.gc_preserve_end(token [[AUG_TOK]])

; CHECK-LABEL: define internal void @diffecopy_memref(ptr nocapture readonly %dst_ref, ptr nocapture %"dst_ref'", ptr nocapture readonly %src_ref, ptr nocapture %"src_ref'"
; CHECK: entry:
; CHECK-DAG: [[REV_DST_SH:%"dst_mem'il_phi"]] = extractvalue {{.+}} %truetape,
; CHECK-DAG: [[REV_SRC_SH:%"src_mem'il_phi"]] = extractvalue {{.+}} %truetape,
; CHECK: invertentry:
; CHECK: [[REV_TOK:%.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %dst_mem, ptr addrspace(10) [[REV_DST_SH]], ptr addrspace(10) %src_mem, ptr addrspace(10) [[REV_SRC_SH]])
; CHECK: call void @llvm.julia.gc_preserve_end(token [[REV_TOK]])
