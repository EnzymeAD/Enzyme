; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Julia 1.14 allocates a Memory with the julia.gc_alloc_memory intrinsic instead
; of jl_alloc_genericmemory (see readonlyorthrow_fresh_memory.ll). Writing its
; data writes memory that did not exist before the call: @fill is fully
; read-only-or-throw, and @fill_ret, which returns the Memory, is only local.

@memty = external addrspace(10) global i8

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_memory(ptr, i64, ptr addrspace(10), i64, i1) #0
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr) #1

define void @fill(ptr %task, ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %nbytes = shl i64 %n, 3
  %m = call ptr addrspace(10) @julia.gc_alloc_memory(ptr %task, i64 %nbytes, ptr addrspace(10) @memty, i64 %n, i1 false)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr addrspace(13) %rooted, align 8
  ret void
}

define ptr addrspace(10) @fill_ret(ptr %task, ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %nbytes = shl i64 %n, 3
  %m = call ptr addrspace(10) @julia.gc_alloc_memory(ptr %task, i64 %nbytes, ptr addrspace(10) @memty, i64 %n, i1 false)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr addrspace(13) %rooted, align 8
  ret ptr addrspace(10) %m
}

attributes #0 = { "enzyme_ReadOnlyOrThrow" }
attributes #1 = { nounwind memory(none) }

; CHECK: define void @fill({{.*}}) #[[FILL:[0-9]+]] {
; CHECK: define ptr addrspace(10) @fill_ret({{.*}}) #[[LOCAL:[0-9]+]] {
; CHECK-DAG: attributes #[[FILL]] = { {{.*}}"enzyme_ReadOnlyOrThrow"{{.*}} }
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }
