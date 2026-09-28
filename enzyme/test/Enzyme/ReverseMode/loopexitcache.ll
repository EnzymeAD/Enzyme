; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; The loops in @mag and @nested are not rotated: the running sum %s is a loop
; header phi that is only used after the loop exits. Only its value from the
; final iteration is needed in the reverse pass, so the augmented forward pass
; should cache a single value (per iteration of any enclosing loop) rather than
; one value per iteration of the loop defining %s.

declare double @llvm.sqrt.f64(double)

define double @mag(ptr noalias %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %body ]
  %s = phi double [ 0.000000e+00, %entry ], [ %add, %body ]
  %cmp = icmp slt i64 %i, %n
  br i1 %cmp, label %body, label %exit

body:
  %gep = getelementptr inbounds double, ptr %x, i64 %i
  %ld = load double, ptr %gep, align 8
  %mul = fmul double %ld, %ld
  %add = fadd double %s, %mul
  %inc = add nuw nsw i64 %i, 1
  br label %loop

exit:
  %r = call double @llvm.sqrt.f64(double %s)
  ret double %r
}

; The inner loop has two exits, neither of which depends on %s.
define void @nested(ptr noalias %out, ptr noalias %x, i64 %n) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %xj.p = getelementptr inbounds double, ptr %x, i64 %j
  %xj = load double, ptr %xj.p, align 8
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %inc, %body ]
  %s = phi double [ %xj, %outer ], [ %add, %body ]
  %cmp = icmp slt i64 %i, %n
  br i1 %cmp, label %check, label %outer.latch

check:
  %gep = getelementptr inbounds double, ptr %x, i64 %i
  %ld = load double, ptr %gep, align 8
  %neg = fcmp olt double %ld, 0.000000e+00
  br i1 %neg, label %outer.latch, label %body

body:
  %mul = fmul double %ld, %ld
  %add = fadd double %s, %mul
  %inc = add nuw nsw i64 %i, 1
  br label %inner

outer.latch:
  %sq = call double @llvm.sqrt.f64(double %s)
  %res = fmul double %xj, %sq
  %o.p = getelementptr inbounds double, ptr %out, i64 %j
  store double %res, ptr %o.p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %cmp2 = icmp slt i64 %j.next, %n
  br i1 %cmp2, label %outer, label %exit

exit:
  ret void
}

define void @normalize(ptr noalias %out2, ptr noalias %out, ptr noalias %x, i64 %n) {
entry:
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %loop ]
  %m = call double @mag(ptr %x, i64 %n)
  call void @nested(ptr %out2, ptr %x, i64 %n)
  %xj.p = getelementptr inbounds double, ptr %x, i64 %j
  %xj = load double, ptr %xj.p, align 8
  %div = fdiv double %xj, %m
  %o.p = getelementptr inbounds double, ptr %out, i64 %j
  store double %div, ptr %o.p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %cmp = icmp slt i64 %j.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

define void @dnormalize(ptr %out2, ptr %dout2, ptr %out, ptr %dout, ptr %x, ptr %dx, i64 %n) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @normalize, metadata !"enzyme_dup", ptr %out2, ptr %dout2, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %x, ptr %dx, i64 %n)
  ret void
}

; For the nested loop, one value per iteration of the outer loop is cached
; (n doubles), instead of a growing per-inner-iteration buffer per outer
; iteration.
; CHECK: define internal ptr @augmented_nested(ptr noalias writeonly captures(none) %out, ptr captures(none) %"out'", ptr noalias readonly captures(none) %x, ptr captures(none) %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %smax = call i64 @llvm.smax.i64(i64 %n, i64 1)
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %smax, 8
; CHECK-NEXT:   %s_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   br label %outer
; CHECK: inner:
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %body ], [ 0, %outer ]
; CHECK-NEXT:   %s = phi double [ %xj, %outer ], [ %add, %body ]
; CHECK-NEXT:   %[[gep:.+]] = getelementptr inbounds double, ptr %s_malloccache, i64 %iv
; CHECK-NEXT:   store double %s, ptr %[[gep]]
; CHECK-NOT: realloc
; CHECK: ret ptr %s_malloccache

; The tape of @mag is the final running sum itself (no per-iteration cache),
; overwritten in the loop header on every iteration.
; CHECK: define internal { double, double } @augmented_mag(ptr noalias readonly captures(none) %x, ptr captures(none) %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double, double }
; CHECK-NEXT:   br label %loop
; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %body ], [ 0, %entry ]
; CHECK-NEXT:   %s = phi double [ 0.000000e+00, %entry ], [ %add, %body ]
; CHECK-NEXT:   store double %s, ptr %0
; CHECK-NOT: malloc
; CHECK: exit:
; CHECK-NEXT:   %r = call double @llvm.sqrt.f64(double %s)
; CHECK: ret { double, double }

; CHECK: define internal void @diffemag(ptr noalias readonly captures(none) %x, ptr captures(none) %"x'", i64 %n, double %differeturn, double %s)
; CHECK-NOT: malloc
; CHECK-NOT: free
; CHECK: invertexit:
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast ueq double %s, 0.000000e+00
; CHECK-NEXT:   %[[sq:.+]] = call fast double @llvm.sqrt.f64(double %s)
