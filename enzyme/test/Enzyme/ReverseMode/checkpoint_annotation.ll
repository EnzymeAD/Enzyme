; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; A loop annotated with __enzyme_set_checkpointing(mode, count), which the
; Clang attribute [[enzyme_checkpointing_enable]] emits (and Enzyme-JAX raises
; into Enzyme-MLIR's loop attributes), is outlined into a checkpointed loop:
; one iteration a step, the induction variable recomputed from the step index,
; run by the reference scheme of the mode with the budget as its config.

@g = global double 0.000000e+00

declare void @__enzyme_set_checkpointing(i64, i64)

define void @run(i64 %n) {
entry:
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @__enzyme_set_checkpointing(i64 2, i64 3)
  %v = load double, ptr @g
  %c = sitofp i64 %i to double
  %w = fadd double %v, %c
  store double %w, ptr @g
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: define void @run(i64 %n)
; CHECK: %ckpt.config = alloca { i64, i32, ptr, i64, ptr }
; CHECK: loop.preheader:
; CHECK-NEXT: %0 = call ptr @__enzyme_checkpoint_builtin(i64 2) #[[inactive:.+]], !enzyme_inactive
; CHECK: store i64 3, ptr %1
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, i64 %n)
; CHECK-NEXT: br label %exit.loopexit
; CHECK-NOT: __enzyme_set_checkpointing(

; CHECK: define internal void @run.ckpt.step(i64 %k, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = mul i64 %k, 1
; CHECK-NEXT:   %1 = add i64 0, %0
; CHECK-NEXT:   br label %loop
; CHECK: loop:
; CHECK-NEXT:   %v = load double, ptr @g
; CHECK-NEXT:   %c = sitofp i64 %1 to double
; CHECK-NEXT:   %w = fadd double %v, %c
; CHECK-NEXT:   store double %w, ptr @g
; CHECK: br i1 %done, label %next, label %next
; CHECK: next:
; CHECK-NEXT:   ret void

; CHECK: declare ptr @__enzyme_checkpoint_builtin(i64) #[[builtin:.+]]

; CHECK: define internal void @enzyme.ckpt.for.run.ckpt.step(i64 "enzyme_inactive" %0, i64 "enzyme_inactive" %1, ptr "enzyme_inactive" %2, ptr "enzyme_inactive" %3, i64 %4) #{{.*}} !enzyme_checkpoint_step
; CHECK: attributes #[[builtin]] = { nounwind willreturn memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" }
; CHECK: attributes #[[inactive]] = { memory(none) "enzyme_inactive" }
