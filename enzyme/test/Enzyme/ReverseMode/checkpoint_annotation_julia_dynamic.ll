; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; Julia state whose type does not describe it, or whose references the loop
; changes, is described by Enzyme.jl as a callback region:
; __enzyme_julia_dynamic_state(value, root, callbacks). The root goes to the
; loop function as a region pointer that keeps its derivative (the callbacks
; get its shadow), with the callbacks as its size and its space all ones.
; A stack slot holding an object reference that the loop leaves is a callback
; region with the callbacks of __enzyme_julia_ref_slots.

declare void @__enzyme_julia_dynamic_state(...)
declare void @__enzyme_julia_ref_slots(ptr, ptr)
declare ptr @julia.get_pgcstack()
declare ptr @julia.pointer_from_objref(ptr addrspace(11))
declare void @julia_step(ptr addrspace(10))

@cb = external global [6 x ptr]
@refcb = external global [6 x ptr]
@ptrcb = external global [6 x ptr]

define ptr addrspace(10) @run(ptr addrspace(10) %m, i64 %n) {
entry:
  %ma = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %root = call ptr @julia.pointer_from_objref(ptr addrspace(11) %ma)
  call void @__enzyme_julia_ref_slots(ptr @refcb, ptr @ptrcb)
  call void (...) @__enzyme_julia_dynamic_state(ptr addrspace(10) %m, ptr %root, ptr @cb)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  call void @julia_step(ptr addrspace(10) %m)
  %f = load ptr addrspace(10), ptr addrspace(11) %ma
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret ptr addrspace(10) %f
}

!0 = distinct !{!0, !1}
!1 = !{!"enzyme.checkpoint", !"revolve", i64 3}

; CHECK: define ptr addrspace(10) @run(
; CHECK: %f.reg2mem = alloca ptr addrspace(10)
; CHECK-NOT: __enzyme_julia_dynamic_state(
; CHECK-NOT: __enzyme_julia_ref_slots(
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, ptr %root, i64 ptrtoint (ptr @cb to i64), ptr %f.reg2mem, i64 ptrtoint (ptr @refcb to i64), ptr addrspace(10) %m, ptr %f.reg2mem, i64 %n)
; CHECK: %f.reload = load ptr addrspace(10), ptr %f.reg2mem

; The roots of callback regions keep their derivatives; the schedule's own
; arguments do not.
; CHECK: define internal void @enzyme.ckpt.for.run.ckpt.step(i64 "enzyme_inactive" %0, i64 "enzyme_inactive" %1, ptr "enzyme_inactive" %2, ptr "enzyme_inactive" %3, ptr %4, i64 "enzyme_inactive" %5, ptr %6, i64 "enzyme_inactive" %7, ptr addrspace(10) %8, ptr %9, i64 %10) #[[loopattrs:.+]] !enzyme_checkpoint_step
; CHECK: attributes #[[loopattrs]] = { {{.*}}"enzyme_checkpoint_region_spaces"="4294967295,4294967295"
