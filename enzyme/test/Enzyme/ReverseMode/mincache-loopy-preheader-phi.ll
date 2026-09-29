; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; The clamp loop's header PHI %current is a loopy select reduction whose
; condition is needed in the reverse pass, so the min-cut cache planner keeps
; %current and then provides its start value %current.ph. That preheader PHI
; merges the argument with the scale loop's exit value %acc.next; it cannot be
; recomputed in the reverse pass, so the planner must cache it rather than
; mark it recomputable.

define double @clamp(double %x, i64 %n, i64 %m, i1 %above) {
entry:
  %skip = icmp eq i64 %n, 0
  br i1 %skip, label %clamp.preheader, label %scale

scale:
  %i = phi i64 [ 0, %entry ], [ %i.next, %scale ]
  %acc = phi double [ %x, %entry ], [ %acc.next, %scale ]
  %acc.next = fmul double %acc, %x
  %i.next = add i64 %i, 1
  %scaled = icmp eq i64 %i.next, %n
  br i1 %scaled, label %clamp.preheader, label %scale

clamp.preheader:
  %current.ph = phi double [ %x, %entry ], [ %acc.next, %scale ]
  br label %clamp

clamp:
  %j = phi i64 [ 0, %clamp.preheader ], [ %j.next, %clamp ]
  %current = phi double [ %current.ph, %clamp.preheader ], [ %next, %clamp ]
  %less = fcmp olt double %current, 0.000000e+00
  %greater = fcmp ogt double %current, 0.000000e+00
  %outside = select i1 %above, i1 %greater, i1 %less
  %next = select i1 %outside, double 0.000000e+00, double %current
  %j.next = add i64 %j, 1
  %clamped = icmp eq i64 %j.next, %m
  br i1 %clamped, label %exit, label %clamp

exit:
  ret double %next
}

define double @dclamp(double %x, i64 %n, i64 %m, i1 %above) {
  %r = call double (...) @__enzyme_autodiff(ptr @clamp, double %x, i64 %n, i64 %m, i1 %above)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; The start value stays a preheader PHI of the primal, and the reverse of the
; clamp loop carries its adjoint back to the scale loop and the argument.

; CHECK: define internal { double } @diffeclamp(double %x, i64 %n, i64 %m, i1 %above, double %differeturn)
; CHECK: clamp.preheader:
; CHECK:   %current.ph = phi double [ %x, %entry ], [ %acc.next, %scale{{.*}} ]
; CHECK: invertclamp:
; CHECK:   %"current.ph'de.0" = phi double
