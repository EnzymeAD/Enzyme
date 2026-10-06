; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; Enzyme.jl describes the Julia values a checkpointed loop uses with
; __enzyme_julia_state(value, ptr, bytes, ...): what can be written through
; them, from their types. Here a struct of two arrays, passed by value and so
; kept in a stack slot, which the step passes to a call that writes the
; arrays' data (Julia marks it readonly: the callee does not write the slot).
; The snapshot holds the slot and both arrays' data; without the marker the
; loop is refused.

declare void @__enzyme_julia_state(...)
declare ptr @julia.get_pgcstack()
declare void @julia_step(ptr addrspace(11) readonly)

define void @run([2 x ptr addrspace(10)] %m, i64 %n) {
entry:
  %slot = alloca [2 x ptr addrspace(10)]
  %u = extractvalue [2 x ptr addrspace(10)] %m, 0
  store ptr addrspace(10) %u, ptr %slot
  %t = extractvalue [2 x ptr addrspace(10)] %m, 1
  %slot1 = getelementptr inbounds [2 x ptr addrspace(10)], ptr %slot, i64 0, i64 1
  store ptr addrspace(10) %t, ptr %slot1
  %s = addrspacecast ptr %slot to ptr addrspace(11)
  %ua = addrspacecast ptr addrspace(10) %u to ptr addrspace(11)
  %ud = load ptr, ptr addrspace(11) %ua
  %ta = addrspacecast ptr addrspace(10) %t to ptr addrspace(11)
  %td = load ptr, ptr addrspace(11) %ta
  call void (...) @__enzyme_julia_state(ptr addrspace(11) %s, ptr %ud, i64 80, ptr %td, i64 80)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @julia_step(ptr addrspace(11) %s)
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"enzyme.checkpoint", !"revolve", i64 3}

; CHECK: define void @run(
; CHECK-NOT: __enzyme_julia_state(
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %{{.+}}, ptr %ckpt.config, ptr %ud, i64 80, ptr %td, i64 80, ptr %slot, i64 16, i64 %n, ptr %slot)
; CHECK: define internal void @run.ckpt.step(i64 %k, i64 %n, ptr %slot)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %s = addrspacecast ptr %slot to ptr addrspace(11)
