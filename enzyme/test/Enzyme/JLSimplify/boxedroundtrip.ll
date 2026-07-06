; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S -opaque-pointers | FileCheck %s

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

; When stored value %v is an instruction that dominates the load %contents,
; the stored value could be in the box, so comparison must not be folded.
define i1 @boxed_roundtrip_inst(ptr %task, ptr addrspace(10) %tag, ptr addrspace(11) %src) {
top:
  %v = load ptr addrspace(10), ptr addrspace(11) %src, align 8
  %box = call noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) %tag)
  %slot = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  store ptr addrspace(10) %v, ptr addrspace(11) %slot, align 8
  %contents = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %cmp = icmp eq ptr addrspace(10) %contents, %v
  ret i1 %cmp
}

; CHECK: define i1 @boxed_roundtrip_inst
; CHECK:   %cmp = icmp eq ptr addrspace(10) %contents, %v
; CHECK-NEXT:   ret i1 %cmp

; When load %contents dominates instruction %after_v, %after_v did not exist
; at the time of %contents load, and box is not captured before %after_v,
; so fold to false is legal.
define i1 @boxed_roundtrip_ld_dominates(ptr %task, ptr addrspace(10) %tag, ptr addrspace(10) %v, ptr addrspace(11) %other_slot) {
top:
  %box = call noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) %tag)
  %slot = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  store ptr addrspace(10) %v, ptr addrspace(11) %slot, align 8
  %contents = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %after_v = load ptr addrspace(10), ptr addrspace(11) %other_slot, align 8
  %cmp = icmp eq ptr addrspace(10) %contents, %after_v
  ret i1 %cmp
}

; CHECK: define i1 @boxed_roundtrip_ld_dominates
; CHECK:   ret i1 false

; Two loads from the exact same allocation load identical values; comparison must not be folded to false.
define i1 @same_alloc_loads(ptr %task, ptr addrspace(10) %tag) {
top:
  %box = call noalias nonnull align 8 dereferenceable(8) ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) %tag)
  %slot = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  %c1 = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %c2 = load ptr addrspace(10), ptr addrspace(11) %slot, align 8
  %cmp = icmp eq ptr addrspace(10) %c1, %c2
  ret i1 %cmp
}

; CHECK: define i1 @same_alloc_loads
; CHECK:   %cmp = icmp eq ptr addrspace(10) %c1, %c2
; CHECK-NEXT:   ret i1 %cmp


