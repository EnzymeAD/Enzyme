; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Julia 1.14's field barrier names each (slot, child) pair it covers. The
; shadow store needs its own barrier, which keeps the slot even when the
; stored child is inactive.

declare void @julia.field_write_barrier.p11(ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...)

define void @set(ptr addrspace(10) %parent, ptr addrspace(10) %child) {
top:
  %slot = addrspacecast ptr addrspace(10) %parent to ptr addrspace(11)
  call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %parent, ptr addrspace(11) %slot, ptr addrspace(10) %child)
  store ptr addrspace(10) %child, ptr addrspace(11) %slot, align 8
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @dactive(ptr addrspace(10) %p, ptr addrspace(10) %dp, ptr addrspace(10) %c, ptr addrspace(10) %dc) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @set, metadata !"enzyme_dup", ptr addrspace(10) %p, ptr addrspace(10) %dp, metadata !"enzyme_dup", ptr addrspace(10) %c, ptr addrspace(10) %dc)
  ret void
}

define void @dconst(ptr addrspace(10) %p, ptr addrspace(10) %dp, ptr addrspace(10) %c) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @set, metadata !"enzyme_dup", ptr addrspace(10) %p, ptr addrspace(10) %dp, metadata !"enzyme_const", ptr addrspace(10) %c)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffeset(
; CHECK: %"slot'ipc" = addrspacecast ptr addrspace(10) %"parent'" to ptr addrspace(11)
; CHECK-NEXT: %slot = addrspacecast ptr addrspace(10) %parent to ptr addrspace(11)
; CHECK-NEXT: call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %"parent'", ptr addrspace(11) %"slot'ipc", ptr addrspace(10) %"child'")
; CHECK-NEXT: call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %parent, ptr addrspace(11) %slot, ptr addrspace(10) %child)
; CHECK-NEXT: store ptr addrspace(10) %"child'", ptr addrspace(11) %"slot'ipc"
; CHECK-NEXT: store ptr addrspace(10) %child, ptr addrspace(11) %slot

; CHECK-LABEL: define internal void @fwddiffeset.1(
; CHECK: %"slot'ipc" = addrspacecast ptr addrspace(10) %"parent'" to ptr addrspace(11)
; CHECK-NEXT: %slot = addrspacecast ptr addrspace(10) %parent to ptr addrspace(11)
; CHECK-NEXT: call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %"parent'", ptr addrspace(11) %"slot'ipc", ptr addrspace(10) %child)
; CHECK-NEXT: call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %parent, ptr addrspace(11) %slot, ptr addrspace(10) %child)
; CHECK-NEXT: store ptr addrspace(10) %child, ptr addrspace(11) %"slot'ipc"
; CHECK-NEXT: store ptr addrspace(10) %child, ptr addrspace(11) %slot
