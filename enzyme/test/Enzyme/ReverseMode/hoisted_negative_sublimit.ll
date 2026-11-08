; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_autodiff(...)

define double @nested(ptr nocapture readonly %x, i64 %outer_n, i64 %m) {
entry:
  %inner_lim_minus_2 = add i64 %m, -2
  %guard = icmp sge i64 %m, 2
  br label %outer.loop

outer.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc = phi double [ 1.0, %entry ], [ %acc.next, %outer.latch ]
  %xi = load double, ptr %x, align 8
  %prod0 = fmul double %acc, %xi
  br i1 %guard, label %inner.loop, label %outer.latch

inner.loop:
  %j = phi i64 [ 0, %outer.loop ], [ %j.next, %inner.loop ]
  %c = phi double [ %prod0, %outer.loop ], [ %c.next, %inner.loop ]
  %c.next = fmul double %c, %xi
  %j.done = icmp eq i64 %j, %inner_lim_minus_2
  %j.next = add nuw nsw i64 %j, 1
  br i1 %j.done, label %outer.latch, label %inner.loop

outer.latch:
  %acc.next = phi double [ %prod0, %outer.loop ], [ %c.next, %inner.loop ]
  %outer.done = icmp eq i64 %i, %outer_n
  %i.next = add nuw nsw i64 %i, 1
  br i1 %outer.done, label %exit, label %outer.loop

exit:
  ret double %acc.next
}

define void @dnested(ptr %x, ptr %dx, i64 %outer_n, i64 %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @nested, metadata !"enzyme_dup", ptr %x, ptr %dx, i64 %outer_n, i64 %m)
  ret void
}

; CHECK-LABEL: define internal void @diffenested(
; CHECK: %[[ADD:.+]] = add nsw i64 %inner_lim_minus_2, 1
; CHECK-NEXT: %[[CMP:.+]] = icmp slt i64 %[[ADD]], 0
; CHECK-NEXT: %[[LIM:.+]] = select i1 %[[CMP]], i64 0, i64 %[[ADD]]
