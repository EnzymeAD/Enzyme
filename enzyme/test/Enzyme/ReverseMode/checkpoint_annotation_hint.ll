; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; __enzyme_ptr_size_hint(ptr, bytes, space) before an annotated loop gives the
; extent of memory the loop writes through a pointer it did not allocate, and
; the memory space it is really in: a snapshot region of %p, 80 bytes, in
; space 1 (a device buffer behind a plain pointer). The hint is then dropped.

declare void @__enzyme_set_checkpointing(i64, i64)
declare void @__enzyme_ptr_size_hint(ptr, i64, i64)

define void @run(ptr %p, i64 %n) {
entry:
  call void @__enzyme_ptr_size_hint(ptr %p, i64 80, i64 1)
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @__enzyme_set_checkpointing(i64 2, i64 4)
  %v = load double, ptr %p
  %w = fmul double %v, %v
  store double %w, ptr %p
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: define void @run(ptr %p, i64 %n)
; CHECK-NOT: __enzyme_ptr_size_hint(
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %{{.*}}, ptr %ckpt.config, ptr %p, i64 80, ptr %p, i64 %n)
; CHECK: define internal void @enzyme.ckpt.for.run.ckpt.step({{.*}}) #[[attrs:.+]] !enzyme_checkpoint_step
; CHECK: attributes #[[attrs]] = { noinline "enzyme_checkpoint"="for" "enzyme_checkpoint_nregions"="1" "enzyme_checkpoint_region_spaces"="1" }
