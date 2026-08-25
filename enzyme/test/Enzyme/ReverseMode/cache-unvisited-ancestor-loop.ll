; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; overwritesToMemoryReadByLoop must see *every* common loop of the load and the
; store as an induction variable before it may report "no overwrite".  Here the
; load and the store share two loops:
;
;   for (i = 0; i < N; i++) {
;     off = offs[i];                      // opaque, varies with i
;     for (j = 0; j < M; j++) {
;       v = A[off + M + 1 + j];           // load
;       s += v * v;
;       A[off + j] = 3.0;                 // store
;     }
;   }
;
; Within one iteration of the outer loop the store range [off, off+M) lies
; strictly below the load range [off+M+1, off+2M+1), so SCEV proves the two do
; not overlap and marks the *inner* loop visited.  The outer loop is never
; visited: `off` is an opaque load, so the store address is not an AddRec of
; the outer loop.  A later outer iteration with off' = off + M + 1 therefore
; overwrites exactly what an earlier iteration read, and the load must be
; cached rather than recomputed in the reverse sweep.

declare double @__enzyme_autodiff(ptr, ...)

@enzyme_const = external global i32

define double @f(ptr %A, ptr %offs, i64 %N, i64 %M) {
entry:
  %cmp = icmp slt i64 %N, 1
  %cmp1 = icmp slt i64 %M, 1
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %cleanup, label %outer.preheader

outer.preheader:                                  ; preds = %entry
  br label %outer

outer:                                            ; preds = %outer.latch, %outer.preheader
  %s.0 = phi double [ 0.000000e+00, %outer.preheader ], [ %s.1.lcssa, %outer.latch ]
  %i = phi i64 [ 0, %outer.preheader ], [ %i.next, %outer.latch ]
  %ec.outer = icmp ne i64 %i, %N
  br i1 %ec.outer, label %outer.body, label %cleanup.loopexit

outer.body:                                       ; preds = %outer
  %offp = getelementptr inbounds i64, ptr %offs, i64 %i
  %off = load i64, ptr %offp, align 8
  br label %inner

inner:                                            ; preds = %inner.body, %outer.body
  %s.1 = phi double [ %s.0, %outer.body ], [ %s.next, %inner.body ]
  %j = phi i64 [ 0, %outer.body ], [ %j.next, %inner.body ]
  %ec.inner = icmp ne i64 %j, %M
  br i1 %ec.inner, label %inner.body, label %outer.latch

inner.body:                                       ; preds = %inner
  %base = getelementptr double, ptr %A, i64 %off
  %hi = getelementptr double, ptr %base, i64 %M
  %hi1 = getelementptr double, ptr %hi, i64 1
  %ldp = getelementptr double, ptr %hi1, i64 %j
  %v = load double, ptr %ldp, align 8
  %stp = getelementptr double, ptr %base, i64 %j
  store double 3.000000e+00, ptr %stp, align 8
  %sq = fmul double %v, %v
  %s.next = fadd double %s.1, %sq
  %j.next = add nuw nsw i64 %j, 1
  br label %inner

outer.latch:                                      ; preds = %inner
  %s.1.lcssa = phi double [ %s.1, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  br label %outer

cleanup.loopexit:                                 ; preds = %outer
  %s.0.lcssa = phi double [ %s.0, %outer ]
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %entry
  %r = phi double [ 0.000000e+00, %entry ], [ %s.0.lcssa, %cleanup.loopexit ]
  ret double %r
}

define double @dsq(ptr %A, ptr %dA, ptr %offs, i64 %N, i64 %M) {
entry:
  %c = load i32, ptr @enzyme_const, align 4
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %A, ptr %dA, i32 %c, ptr %offs, i64 %N, i64 %M)
  ret double %r
}

; CHECK: define internal void @diffef(ptr {{.*}}%A, ptr {{.*}}%"A'", ptr {{.*}}%offs, i64 %N, i64 %M, double %differeturn)

; The load must be cached: performed in the forward sweep and spilled ...
; CHECK: %v_malloccache = tail call noalias nonnull ptr @malloc(
; CHECK: %v = load double, ptr %ldp, align 8
; CHECK: store double %v, ptr %{{.+}}, align 8

; ... then read back from the cache in the reverse sweep, rather than being
; recomputed out of %A after the store has clobbered it.
; CHECK: %[[GEP:.+]] = getelementptr inbounds double, ptr %v_malloccache, i64 %{{.+}}
; CHECK-NEXT: %{{.+}} = load double, ptr %[[GEP]], align 8
; CHECK-NOT: %v_unwrap
