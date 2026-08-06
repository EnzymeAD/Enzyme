; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; The start value of a loopy reduction is needed in the reverse pass, but is
; only discovered after the min cut has been computed. Here that start value
; (%seed) is a load which is not legal to recompute, so it must keep the
; explicit cache decision made for it rather than being marked as recomputed.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@enzyme_primal_return = external global ptr
@enzyme_dup = external global ptr
@enzyme_const = external global ptr

define hidden double @wrap(ptr nocapture readonly %diag, ptr nocapture writeonly %out) {
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

define double @entry(ptr %diag, ptr %ddiag, ptr %out) {
  %r = tail call double (...) @__enzyme_autodiff(ptr @wrap, ptr @enzyme_primal_return, ptr @enzyme_dup, ptr %diag, ptr %ddiag, ptr @enzyme_const, ptr %out)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffewrap(ptr {{.*}}%diag, ptr {{.*}}%"diag'", ptr {{.*}}%out, double %differeturn)
; The start value is loaded once in the primal, and its adjoint accumulated in
; the reverse; it is never recomputed there.
; CHECK: %seed = load double, ptr %diag
; CHECK: invertstart:
; CHECK-NOT: load double, ptr %diag
