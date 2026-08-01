; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; A loopy reduction PHI whose preheader incoming value is an instruction (here a
; load from an active argument) rather than a constant.
;
; pushLoopyPHIPreheader must not add such a value to the min-cut recompute graph
; unless it is legal to recompute, since computeMinCache later assumes exactly
; that of every Intermediate the min-cut declined to cache. Adding it
; unconditionally aborted with
;   GradientUtils.cpp: void GradientUtils::computeMinCache():
;     Assertion `legalRecompute(V, Available2, nullptr)' failed.

declare double @__enzyme_autodiff(...)

@enzyme_dup = external global i32
@enzyme_const = external global i32

define double @wrap(ptr nocapture readonly %diag, ptr nocapture writeonly %out) {
start:
  %seed = load double, ptr %diag, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop ]
  %gu = phi double [ %seed, %start ], [ %gu.next, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %p = getelementptr inbounds double, ptr %diag, i64 %i
  %v = load double, ptr %p, align 8
  %cmp = fcmp ogt double %gu, %v
  %vplus = fadd double %v, 0.000000e+00
  %gu.next = select i1 %cmp, double %gu, double %vplus
  %done = icmp eq i64 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  %gu.final = phi double [ %gu.next, %loop ]
  %isneg = fcmp ogt double %gu.final, 0.000000e+00
  %sigma = select i1 %isneg, double 0.000000e+00, double %gu.final
  %slot = getelementptr inbounds i8, ptr %out, i64 56
  store double %sigma, ptr %slot, align 8
  ret double 0.000000e+00
}

define double @caller(ptr %diag, ptr %ddiag, ptr %out) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @wrap, ptr @enzyme_dup, ptr %diag, ptr %ddiag, ptr @enzyme_const, ptr %out)
  ret double %r
}

; CHECK: define internal void @diffewrap(

; The adjoint of the preheader seed load must be accumulated into the shadow of
; the active argument.
; CHECK: invertstart:
; CHECK:   %[[seedde:.+]] = load double, {{.+}} %"seed'de"
; CHECK:   store double 0.000000e+00, {{.+}} %"seed'de"
; CHECK:   %[[prev:.+]] = load double, {{.+}} %"diag'"
; CHECK:   %[[acc:.+]] = fadd fast double %[[prev]], %[[seedde]]
; CHECK:   store double %[[acc]], {{.+}} %"diag'"
