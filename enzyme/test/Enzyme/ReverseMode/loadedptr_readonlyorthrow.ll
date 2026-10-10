; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Like loadedptr_nocache.ll, but the array is a noalias argument and after the
; call @f calls @check, which only writes memory on paths that throw
; (enzyme_ReadOnlyOrThrow), and @mk, which additionally writes only its sret
; argument, an uncaptured noalias argument (enzyme_LocalReadOnlyOrThrow). If
; either throws the reverse pass never runs, and otherwise neither writes
; memory @sqdist may read, so the data @sqdist reads need not be cached.

declare void @__enzyme_autodiff(...)
declare void @check(i64) nofree "enzyme_ReadOnlyOrThrow" "enzyme_inactive"

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

define void @f(ptr noalias nocapture %out, ptr noalias %arr, ptr noalias nocapture %buf, double %m) {
entry:
  %r = call double @sqdist(ptr %arr, double %m)
  call void @check(i64 1)
  call void @mk(ptr sret(i64) %buf, i64 2)
  store double %r, ptr %out
  ret void
}

define void @test(ptr %out, ptr %dout, ptr %arr, ptr %buf, double %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %arr, metadata !"enzyme_const", ptr %buf, double %m)
  ret void
}

; No tape is passed from the augmented forward pass to the reverse pass.
; CHECK: define internal { double } @diffef(
; CHECK: %r = call fast double @augmented_sqdist(ptr %arr, double %m)
; CHECK: call {{(fast )?}}{ double } @diffesqdist(ptr %arr, double %m, double %
; CHECK: }

; CHECK: define internal double @augmented_sqdist(ptr nocapture readonly %arr, double %m)
; CHECK-NOT: malloc
; CHECK: }

; CHECK: define internal { double } @diffesqdist(ptr nocapture readonly %arr, double %m, double %differeturn)
; CHECK-NOT: malloc
; CHECK: %y_unwrap = load double
; CHECK: }
