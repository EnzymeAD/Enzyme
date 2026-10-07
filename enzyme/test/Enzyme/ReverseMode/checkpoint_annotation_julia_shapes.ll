; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; Shapes of Julia loops the outliner handles:
; - a floating-point value the loop uses but does not compute (%nu) goes to
;   the step by reference, through a stack slot, as the driver takes active
;   arguments by reference only;
; - a value picked by a phi in the latch (%v) and used after the loop goes
;   through a stack slot, as other values used after the loop do;
; - an error path the code before the loop shares (%throw) is copied into
;   the step, and stays for the code before the loop.

@g = global double 0.000000e+00

declare void @julia_throw(i64) noreturn

define double @run(double %nu, i64 %n, i1 %bad) {
entry:
  br i1 %bad, label %throw, label %pre

pre:
  br label %loop

loop:
  %i = phi i64 [ 0, %pre ], [ %i.next, %latch ]
  %x = load double, ptr @g
  %neg = fcmp olt double %x, 0.000000e+00
  br i1 %neg, label %throw, label %body

body:
  %c = fcmp ogt double %x, 1.000000e+00
  br i1 %c, label %big, label %latch

big:
  %h = fmul double %x, 5.000000e-01
  br label %latch

latch:
  %v = phi double [ %x, %body ], [ %h, %big ]
  %w = fmul double %v, %nu
  store double %w, ptr @g
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

throw:
  %code = phi i64 [ 1, %entry ], [ %i, %loop ]
  call void @julia_throw(i64 %code)
  unreachable

exit:
  ret double %v
}

!0 = distinct !{!0, !1}
!1 = !{!"enzyme.checkpoint", !"revolve", i64 3}

; CHECK: define double @run(double %nu, i64 %n, i1 %bad)
; CHECK: %v.reg2mem = alloca double
; CHECK: %nu.byref = alloca double
; CHECK: pre:
; CHECK-NEXT: store double %nu, ptr %nu.byref
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, ptr %v.reg2mem, i64 8, ptr %nu.byref, i64 8, ptr %v.reg2mem, ptr %nu.byref, i64 %n)
; The shared error path stays, for the code before the loop only.
; CHECK: throw:
; CHECK-NEXT: call void @julia_throw(i64 1)
; CHECK: exit:
; CHECK-NEXT: %v.reload = load double, ptr %v.reg2mem
; CHECK-NEXT: ret double %v.reload

; CHECK: define internal void @run.ckpt.step(i64 %k, ptr %v.reg2mem, ptr %nu.byref, i64 %n)
; CHECK: latch:
; CHECK-NEXT: %v = phi double [ %x, %body ], [ %h, %big ]
; CHECK-NEXT: store double %v, ptr %v.reg2mem
; CHECK-NEXT: %v.reload1 = load double, ptr %v.reg2mem
; CHECK-NEXT: %nu.ld = load double, ptr %nu.byref
; CHECK-NEXT: %w = fmul double %v.reload1, %nu.ld
; CHECK: throw:
; CHECK-NEXT: %code = phi i64 [ %1, %throw.loopexit ]
; CHECK-NEXT: call void @julia_throw(i64 %code)
