; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; @f overwrites %slot, a local alloca passed to @sqdist, after the call, so
; @sqdist must cache the pointer it loads from %slot. The data that pointer
; points to is not written, so the values loaded through it need not be cached.

declare void @__enzyme_autodiff(...)

define internal double @sqdist(ptr %slot, double %m) {
entry:
  %data = load ptr, ptr %slot
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
  %done = icmp eq i64 %inc, 4
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @f(ptr noalias nocapture %out, ptr %s, ptr %other, double %m) {
entry:
  %slot = alloca ptr
  %arr = load ptr, ptr %s
  store ptr %arr, ptr %slot
  %r = call double @sqdist(ptr nocapture %slot, double %m)
  store ptr %other, ptr %slot
  store double %r, ptr %out
  ret void
}

define void @test(ptr %out, ptr %dout, ptr %s, ptr %other, double %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %s, metadata !"enzyme_const", ptr %other, double %m)
  ret void
}

; CHECK: define internal { ptr, double } @augmented_sqdist(ptr nocapture readonly %slot, double %m)
; CHECK-NOT: malloc
; CHECK: %data = load ptr, ptr %slot
; CHECK-NEXT: store ptr %data
; CHECK: define internal { double } @diffesqdist(ptr nocapture readonly %slot, double %m, double %differeturn, ptr %data)
; CHECK-NOT: malloc
; CHECK: %p_unwrap = getelementptr inbounds double, ptr %data
; CHECK-NEXT: %y_unwrap = load double, ptr %p_unwrap
