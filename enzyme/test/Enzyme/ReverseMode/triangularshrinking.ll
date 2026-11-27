; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -early-cse -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg,early-cse,%simplifycfg,instsimplify)" -S | FileCheck %s

; The inner trip count shrinks with the outer induction variable, so the cache
; is sized by the inner trip count of the first outer iteration and allocated
; once. See triangular.ll for a growing inner trip count.

define double @get(double* nocapture %x, i64 %i) noinline nounwind {
entry:
  %p = getelementptr inbounds double, double* %x, i64 %i
  %v = load double, double* %p, align 8
  ret double %v
}

; for i in 0..n-1, for j in i+1..n-1: the inner trip count shrinks.
define double @g(double* nocapture %x, i64 %n) {
entry:
  %nonempty = icmp sgt i64 %n, 0
  br i1 %nonempty, label %outer, label %exit

outer:
  %i = phi i64 [ %i_inc, %outer.latch ], [ 0, %entry ]
  %outersum = phi double [ %sum.latch, %outer.latch ], [ 0.000000e+00, %entry ]
  %i_inc = add nuw nsw i64 %i, 1
  %guard = icmp slt i64 %i_inc, %n
  br i1 %guard, label %inner, label %outer.latch

inner:
  %j = phi i64 [ %j_inc, %inner ], [ %i_inc, %outer ]
  %innersum = phi double [ %innersum.next, %inner ], [ %outersum, %outer ]
  %v = call double @get(double* %x, i64 %j)
  %mul = fmul double %v, %v
  %innersum.next = fadd double %mul, %innersum
  %j_inc = add nuw nsw i64 %j, 1
  %inner.done = icmp eq i64 %j_inc, %n
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %sum.latch = phi double [ %outersum, %outer ], [ %innersum.next, %inner ]
  %outer.done = icmp eq i64 %i_inc, %n
  br i1 %outer.done, label %exit, label %outer

exit:
  %res = phi double [ 0.000000e+00, %entry ], [ %sum.latch, %outer.latch ]
  ret double %res
}

define void @dsumsquare(double* %x, double* %xp, i64 %n) {
entry:
  %0 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @g to i8*), double* %x, double* %xp, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffeg(double* nocapture %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK: outer.preheader:
; CHECK-NEXT:   %[[nm2:.+]] = add nsw i64 %n, -2
; CHECK-NEXT:   %[[gsmax:.+]] = call i64 @llvm.smax.i64(i64 %[[nm2]], i64 0)
; CHECK-NEXT:   %[[ginner:.+]] = add nuw nsw i64 %[[gsmax]], 1
; CHECK-NEXT:   %[[gtotal:.+]] = mul nuw nsw i64 %[[ginner]], %n
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[gtotal]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NOT: @malloc
; CHECK: }
