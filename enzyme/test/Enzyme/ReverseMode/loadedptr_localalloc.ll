; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Like loadedptr_nocache.ll, but after the call @f writes to memory it
; allocated itself (never captured, later freed) and starts the lifetime of a
; local alloca. Neither can be reached through the pointers @sqdist loads, so
; the data @sqdist reads need not be cached.

declare void @__enzyme_autodiff(...)
declare noalias ptr @malloc(i64)
declare void @free(ptr nocapture)
declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)

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
  %tmp = call noalias ptr @malloc(i64 8)
  %buf = alloca double
  %arr = load ptr, ptr %s
  %r = call double @sqdist(ptr %arr, double %m)
  store double %r, ptr %tmp
  call void @llvm.lifetime.start.p0(i64 8, ptr %buf)
  %r2 = load double, ptr %tmp
  store double %r2, ptr %buf
  %r3 = load double, ptr %buf
  call void @llvm.lifetime.end.p0(i64 8, ptr %buf)
  store double %r3, ptr %out
  call void @free(ptr %tmp)
  ret void
}

define void @test(ptr %out, ptr %dout, ptr %s, double %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %s, double %m)
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
