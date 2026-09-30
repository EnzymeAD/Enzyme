; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; In combined mode a load from an array alloca inside a loop may be cached by
; copying the whole alloca before the loop (EnzymeLoopInvariantCache). A
; lookup of such a load at a given scope must also look up the load's index
; at that scope. It did so at the insertion block instead.
;
; Here the reverse pass of %loop needs the cache of %xv, whose size depends on
; %n. Recomputing %n in the reverse pass unwraps %blank.next, which looks up
; %ch (scoped in %scan). Looking up the index %i at the insertion block, which
; corresponds to %loop, made a LCSSA phi of %i in %loop; caching that phi
; needs the limit of %loop, which again unwraps %blank.next: the lookups
; recursed until the stack overflowed.

@name = global [16 x i8] zeroinitializer
@name2 = global [16 x i8] zeroinitializer
@nglob = global i64 8
@seen = global i64 0

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @__enzyme_autodiff(...)

define void @f(ptr %x, i1 %flag) {
entry:
  %buf = alloca [16 x i8], align 1
  br i1 %flag, label %start, label %ret

start:
  call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr @name, i64 16, i1 false)
  %first = load i8, ptr %buf, align 1
  %c0 = icmp eq i8 %first, 32
  br label %scan

scan:
  %blank = phi i1 [ %c0, %start ], [ %blank.next, %latch ]
  %i = phi i64 [ 1, %start ], [ %i.next, %latch ]
  br i1 %blank, label %check, label %latch

check:
  %p = getelementptr [1 x i8], ptr %buf, i64 %i
  %ch = load i8, ptr %p, align 1
  %isb = icmp eq i8 %ch, 32
  br label %latch

latch:
  %blank.next = phi i1 [ %isb, %check ], [ %blank, %scan ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, 16
  br i1 %done, label %exit, label %scan

exit:
  br i1 %blank.next, label %merge, label %nonblank

nonblank:
  store i64 1, ptr @seen, align 8
  %n1 = load i64, ptr @nglob, align 8
  br label %merge

merge:
  %n = phi i64 [ %n1, %nonblank ], [ 4, %exit ]
  br label %loop

loop:
  %k = phi i64 [ 0, %merge ], [ %k.next, %loop ]
  %xp = getelementptr inbounds double, ptr %x, i64 %k
  %xv = load double, ptr %xp, align 8
  %sq = fmul double %xv, %xv
  store double %sq, ptr %xp, align 8
  %k.next = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %k.next, %n
  br i1 %kdone, label %end, label %loop

end:
  call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr @name2, i64 16, i1 false)
  br label %ret

ret:
  ret void
}

define void @test(ptr %x, ptr %dx, i1 %flag) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i1 %flag)
  ret void
}

; The alloca is copied once, before the scan loop.
; CHECK-LABEL: define internal void @diffef(
; CHECK: start:
; CHECK: %ch_malloccache = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) ptr @malloc(i64 16)
; CHECK: store [16 x i8] %{{.*}}, ptr %{{.*}}, align 16
; CHECK: scan:

; No value of the scan loop is made available inside %loop.
; CHECK: loop:
; CHECK-NOT: manual_lcssa
; CHECK: end:
