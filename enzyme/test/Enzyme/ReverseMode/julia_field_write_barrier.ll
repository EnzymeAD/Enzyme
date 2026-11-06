; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Julia 1.14 (julia#62737) emits field-aware write barriers
; `julia.field_write_barrier.p11/.p13(parent, slot, child, ...)` and
; whole-object barriers `julia.object_write_barrier(parent, children...)`.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:10:11:12:13"

declare void @julia.field_write_barrier.p11(ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...)

declare void @julia.field_write_barrier.p13(ptr addrspace(10), ptr addrspace(13), ptr addrspace(10), ...)

declare void @julia.object_write_barrier(ptr addrspace(10), ...)

define void @setfields(ptr addrspace(10) %parent, ptr addrspace(10) %a, ptr addrspace(10) %b) {
entry:
  %p11 = addrspacecast ptr addrspace(10) %parent to ptr addrspace(11)
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %p11, i64 8
  store ptr addrspace(10) %a, ptr addrspace(11) %slot1, align 8
  %slot2 = getelementptr inbounds i8, ptr addrspace(11) %p11, i64 16
  store ptr addrspace(10) %b, ptr addrspace(11) %slot2, align 8
  call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %parent, ptr addrspace(11) %slot1, ptr addrspace(10) %a, ptr addrspace(11) %slot2, ptr addrspace(10) %b)
  %p13 = addrspacecast ptr addrspace(10) %parent to ptr addrspace(13)
  %slot3 = getelementptr inbounds i8, ptr addrspace(13) %p13, i64 24
  store ptr addrspace(10) %a, ptr addrspace(13) %slot3, align 8
  call void (ptr addrspace(10), ptr addrspace(13), ptr addrspace(10), ...) @julia.field_write_barrier.p13(ptr addrspace(10) %parent, ptr addrspace(13) %slot3, ptr addrspace(10) %a)
  %slot4 = getelementptr inbounds i8, ptr addrspace(11) %p11, i64 32
  store ptr addrspace(10) %b, ptr addrspace(11) %slot4, align 8
  call void (ptr addrspace(10), ...) @julia.object_write_barrier(ptr addrspace(10) %parent, ptr addrspace(10) %b)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test(ptr addrspace(10) %parent, ptr addrspace(10) %dparent, ptr addrspace(10) %a, ptr addrspace(10) %da, ptr addrspace(10) %b, ptr addrspace(10) %db) {
entry:
  call void (...) @__enzyme_autodiff(ptr @setfields, metadata !"enzyme_dup", ptr addrspace(10) %parent, ptr addrspace(10) %dparent, metadata !"enzyme_dup", ptr addrspace(10) %a, ptr addrspace(10) %da, metadata !"enzyme_dup", ptr addrspace(10) %b, ptr addrspace(10) %db)
  ret void
}

; CHECK: define internal void @diffesetfields(
; CHECK-NEXT: entry:
; CHECK:   call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %"parent'", ptr addrspace(11) %"slot1'ipg", ptr addrspace(10) %"a'", ptr addrspace(11) %"slot2'ipg", ptr addrspace(10) %"b'")
; CHECK:   call void (ptr addrspace(10), ptr addrspace(11), ptr addrspace(10), ...) @julia.field_write_barrier.p11(ptr addrspace(10) %parent, ptr addrspace(11) %slot1, ptr addrspace(10) %a, ptr addrspace(11) %slot2, ptr addrspace(10) %b)
; CHECK:   call void (ptr addrspace(10), ptr addrspace(13), ptr addrspace(10), ...) @julia.field_write_barrier.p13(ptr addrspace(10) %"parent'", ptr addrspace(13) %"slot3'ipg", ptr addrspace(10) %"a'")
; CHECK:   call void (ptr addrspace(10), ptr addrspace(13), ptr addrspace(10), ...) @julia.field_write_barrier.p13(ptr addrspace(10) %parent, ptr addrspace(13) %slot3, ptr addrspace(10) %a)
; CHECK:   call void (ptr addrspace(10), ...) @julia.object_write_barrier(ptr addrspace(10) %"parent'", ptr addrspace(10) %"b'")
; CHECK:   call void (ptr addrspace(10), ...) @julia.object_write_barrier(ptr addrspace(10) %parent, ptr addrspace(10) %b)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
