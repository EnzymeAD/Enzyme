; RUN: if [ %llvmver -ge 17 ]; then not %opt < %s %newLoadEnzyme -passes="enzyme" -S -o /dev/null 2>&1 | FileCheck %s; fi

; A loop annotated for checkpointing that writes through a pointer of unknown
; extent cannot be snapshotted: it is an error, not a silently wrong gradient.

declare void @__enzyme_set_checkpointing(i64, i64)

define void @run(ptr %p, i64 %n) {
entry:
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @__enzyme_set_checkpointing(i64 1, i64 -1)
  %v = load double, ptr %p
  %w = fmul double %v, %v
  store double %w, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: a checkpointed loop writes through %p, whose extent is not known before it; give it with __enzyme_ptr_size_hint, or the regions with __enzyme_checkpoint_for
