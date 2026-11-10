; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Like loadedptr_readonlyorthrow.ll, but the sret argument @mk writes is not
; local memory of @f: it is the length field of the array @sqdist reads. @mk
; may thus overwrite what @sqdist read, so the values @sqdist loads through
; the array must be cached.

declare void @__enzyme_autodiff(...)

define internal void @mk(ptr nocapture writeonly sret(i64) %p, i64 %v) "enzyme_LocalReadOnlyOrThrow" "enzyme_inactive" {
entry:
  store i64 %v, ptr %p
  ret void
}

define internal double @sqdist(ptr %arr, double %m) {
entry:
  %data = load ptr, ptr %arr
  %np = getelementptr inbounds i8, ptr %arr, i64 8
  %n = load i64, ptr %np
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %data, i64 %i
  %y = load double, ptr %p
  %d = fsub double %y, %m
  %sq = fmul double %d, %d
  %acc.next = fadd double %acc, %sq
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @f(ptr noalias nocapture %out, ptr %s, double %m) {
entry:
  %arr = load ptr, ptr %s
  %r = call double @sqdist(ptr %arr, double %m)
  %np = getelementptr inbounds i8, ptr %arr, i64 8
  call void @mk(ptr sret(i64) %np, i64 2)
  store double %r, ptr %out
  ret void
}

define void @test(ptr %out, ptr %dout, ptr %s, double %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %s, double %m)
  ret void
}

; The data @sqdist loads is cached in the augmented forward pass.
; CHECK: define internal { { i64, ptr }, double } @augmented_sqdist(ptr nocapture readonly %arr, double %m)
; CHECK: malloc
; CHECK: }
