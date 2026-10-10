; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Read-only-or-throw is only stated through function and call-site attributes.
; Instruction metadata of the same name is not a way to exempt a write: a
; function storing through its argument is not read-only-or-throw, whatever
; metadata the store or call carries.

declare void @setfirst(ptr, double)

define void @store_md(ptr %x, double %v) {
entry:
  store double %v, ptr %x, align 8, !enzyme_ReadOnlyOrThrow !0
  ret void
}

define void @call_md(ptr %x, double %v) {
entry:
  call void @setfirst(ptr %x, double %v), !enzyme_LocalReadOnlyOrThrow !0
  ret void
}

!0 = !{}

; The store keeps its argument writable, and neither function is marked.
; CHECK: define void @store_md(ptr {{.*}}writeonly %x, double %v)
; CHECK: define void @call_md(ptr %x, double %v)
; CHECK-NOT: "enzyme_{{(Local)?}}ReadOnlyOrThrow"
