; RUN: if [ %llvmver -ge 16 ]; then not %opt < %s %newLoadEnzyme -passes="enzyme" -S -o /dev/null 2>&1 | FileCheck %s; fi

; The loop of checkpoint_annotation_julia_state.ll without the marker: the
; callee may write the arrays the slot holds, whose extent nothing gives, so
; the loop is refused rather than checkpointed without them.
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

; CHECK: a checkpointed loop writes through a pointer it loads from %slot, whose extent is not known before it
