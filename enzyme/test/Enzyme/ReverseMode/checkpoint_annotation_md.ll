; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; A loop annotated with loop metadata !{!"enzyme.checkpoint", mode, count},
; as Julia's Expr(:loopinfo, (Symbol("enzyme.checkpoint"), :revolve, 3))
; spells it, is checkpointed like one annotated with
; __enzyme_set_checkpointing(mode, count). The entry is removed from the
; loop's metadata, and other entries are kept.

@g = global double 0.000000e+00
@h = global double 0.000000e+00

define void @run(i64 %n) {
entry:
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = load double, ptr @g
  %c = sitofp i64 %i to double
  %w = fadd double %v, %c
  store double %w, ptr @g
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

define void @periodic(i64 %n) {
entry:
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = load double, ptr @h
  %w = fmul double %v, %v
  store double %w, ptr @h
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"enzyme.checkpoint", !"revolve", i64 3}
!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.mustprogress"}
!4 = !{!"enzyme.checkpoint", !"periodic"}

; CHECK: define void @run(i64 %n)
; CHECK: loop.preheader:
; CHECK-NEXT: %0 = call ptr @__enzyme_checkpoint_builtin(i64 2) #[[inactive:.+]], !enzyme_inactive
; CHECK: store i64 3, ptr %1
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, i64 %n)
; CHECK-NOT: enzyme.checkpoint"

; CHECK: define void @periodic(i64 %n)
; CHECK: loop.preheader:
; CHECK-NEXT: %0 = call ptr @__enzyme_checkpoint_builtin(i64 1) #[[inactive]], !enzyme_inactive
; CHECK: call void @enzyme.ckpt.for.periodic.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, i64 %n)

; CHECK: define internal void @run.ckpt.step(i64 %k, i64 %n)
; CHECK: store double %w, ptr @g
; CHECK: define internal void @periodic.ckpt.step(i64 %k, i64 %n)
; CHECK: store double %w, ptr @h

; CHECK-NOT: !{!"enzyme.checkpoint"
