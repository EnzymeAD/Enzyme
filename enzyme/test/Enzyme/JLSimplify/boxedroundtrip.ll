; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

; A pointer stored into a fresh allocation (e.g. a Core.Box for a captured
; variable) and loaded back IS the stored pointer; the comparison against it
; must not be folded away.
define i1 @boxed_roundtrip(ptr %task, ptr addrspace(10) %tag, ptr addrspace(10) %v) {
top:
  %box = call noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) %tag)
  %slot = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  store ptr addrspace(10) %v, ptr addrspace(11) %slot, align 8
  %contents = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %cmp = icmp eq ptr addrspace(10) %contents, %v
  ret i1 %cmp
}

; CHECK: define i1 @boxed_roundtrip
; CHECK:   %cmp = icmp eq ptr addrspace(10) %contents, %v
; CHECK-NEXT:   ret i1 %cmp

; Without a store between the allocation and the load, the loaded value is
; undef (or a fresh allocator-installed pointer), so the fold remains legal.
define i1 @fresh_load(ptr %task, ptr addrspace(10) %tag, ptr addrspace(10) %v) {
top:
  %box = call noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) %tag)
  %slot = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  %contents = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %cmp = icmp eq ptr addrspace(10) %contents, %v
  ret i1 %cmp
}

; CHECK: define i1 @fresh_load
; CHECK:   ret i1 false
