; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

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

; The load must be cached: performed in the forward sweep, spilled into a cache
; allocation, and read back in the reverse sweep -- rather than recomputed out of
; %A after the store has clobbered it.  Without the fix the reverse sweep contains
; %v_unwrap instructions reloading from %A, and no cache at all.
;
; The two cache allocations are matched by capture rather than by name: Enzyme
; derives their names from the cached values on LLVM >= 16 (%v_malloccache,
; %off_malloccache) and falls back to generic ones on LLVM 15 (%malloccall,
; %malloccall15).  The rest of the function is identical on every version.

; CHECK: define internal void @diffef(ptr nocapture %A, ptr nocapture %"A'", ptr nocapture readonly %offs, i64 %N, i64 %M, double %differeturn) #1 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp slt i64 %N, 1
; CHECK-NEXT:   %cmp1 = icmp slt i64 %M, 1
; CHECK-NEXT:   %or.cond = or i1 %cmp, %cmp1
; CHECK-NEXT:   br i1 %or.cond, label %invertcleanup, label %outer.preheader

; CHECK: outer.preheader:
; CHECK-NEXT:   %0 = add nsw i64 %N, 1
; CHECK-NEXT:   %1 = add nsw i64 %M, 1
; CHECK-NEXT:   %2 = mul nuw nsw i64 %1, %0
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %2, 8
; CHECK-NEXT:   %[[vcache:.+]] = tail call noalias nonnull ptr @malloc(i64 %mallocsize), !enzyme_cache_alloc !0
; CHECK-NEXT:   %[[msize2:.+]] = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %[[offcache:.+]] = tail call noalias nonnull ptr @malloc(i64 %[[msize2]]), !enzyme_cache_alloc !2
; CHECK-NEXT:   br label %outer

; CHECK: outer:
; CHECK-NEXT:   %iv = phi i64 [ 0, %outer.preheader ], [ %iv.next, %inner ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %ec.outer = icmp ne i64 %iv, %N
; CHECK-NEXT:   br i1 %ec.outer, label %outer.body, label %invertcleanup

; CHECK: outer.body:
; CHECK-NEXT:   %offp = getelementptr inbounds i64, ptr %offs, i64 %iv
; CHECK-NEXT:   %off = load i64, ptr %offp, align 8, !alias.scope !4, !noalias !7
; CHECK-NEXT:   %3 = getelementptr inbounds i64, ptr %[[offcache]], i64 %iv
; CHECK-NEXT:   store i64 %off, ptr %3, align 8, !invariant.group !9
; CHECK-NEXT:   br label %inner

; CHECK: inner:
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %inner.body ], [ 0, %outer.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %ec.inner = icmp ne i64 %iv1, %M
; CHECK-NEXT:   br i1 %ec.inner, label %inner.body, label %outer

; CHECK: inner.body:
; CHECK-NEXT:   %base = getelementptr double, ptr %A, i64 %off
; CHECK-NEXT:   %hi = getelementptr double, ptr %base, i64 %M
; CHECK-NEXT:   %hi1 = getelementptr double, ptr %hi, i64 1
; CHECK-NEXT:   %ldp = getelementptr double, ptr %hi1, i64 %iv1
; CHECK-NEXT:   %v = load double, ptr %ldp, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %stp = getelementptr double, ptr %base, i64 %iv1
; CHECK-NEXT:   store double 3.000000e+00, ptr %stp, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %4 = mul nuw nsw i64 %iv, %1
; CHECK-NEXT:   %5 = add nuw nsw i64 %iv1, %4
; CHECK-NEXT:   %6 = getelementptr inbounds double, ptr %[[vcache]], i64 %5
; CHECK-NEXT:   store double %v, ptr %6, align 8, !invariant.group !15
; CHECK-NEXT:   br label %inner

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertouter.preheader:
; CHECK-NEXT:   tail call void @free(ptr nonnull %v_cache.0), !enzyme_cache_free !0
; CHECK-NEXT:   tail call void @free(ptr nonnull %off_cache.0), !enzyme_cache_free !2
; CHECK-NEXT:   br label %invertentry

; CHECK: invertouter:
; CHECK-NEXT:   %"v'de.0" = phi double [ %"v'de.1", %invertinner ], [ 0.000000e+00, %invertcleanup ]
; CHECK-NEXT:   %"sq'de.0" = phi double [ %"sq'de.1", %invertinner ], [ 0.000000e+00, %invertcleanup ]
; CHECK-NEXT:   %"s.next'de.0" = phi double [ %12, %invertinner ], [ 0.000000e+00, %invertcleanup ]
; CHECK-NEXT:   %"s.0'de.0" = phi double [ %14, %invertinner ], [ %30, %invertcleanup ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %9, %invertinner ], [ %N, %invertcleanup ]
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %8 = select fast i1 %7, double 0.000000e+00, double %"s.0'de.0"
; CHECK-NEXT:   br i1 %7, label %invertouter.preheader, label %incinvertouter

; CHECK: incinvertouter:
; CHECK-NEXT:   %9 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertinner

; CHECK: invertinner:
; CHECK-NEXT:   %"v'de.1" = phi double [ %"v'de.0", %incinvertouter ], [ 0.000000e+00, %incinvertinner ]
; CHECK-NEXT:   %"sq'de.1" = phi double [ %"sq'de.0", %incinvertouter ], [ 0.000000e+00, %incinvertinner ]
; CHECK-NEXT:   %"s.next'de.1" = phi double [ %"s.next'de.0", %incinvertouter ], [ 0.000000e+00, %incinvertinner ]
; CHECK-NEXT:   %"s.1'de.1" = phi double [ %8, %incinvertouter ], [ %12, %incinvertinner ]
; CHECK-NEXT:   %"s.0'de.1" = phi double [ 0.000000e+00, %incinvertouter ], [ %14, %incinvertinner ]
; CHECK-NEXT:   %"iv1'ac.1" = phi i64 [ %M, %incinvertouter ], [ %15, %incinvertinner ]
; CHECK-NEXT:   %10 = icmp eq i64 %"iv1'ac.1", 0
; CHECK-NEXT:   %11 = fadd fast double %"s.next'de.1", %"s.1'de.1"
; CHECK-NEXT:   %12 = select fast i1 %10, double %"s.next'de.1", double %11
; CHECK-NEXT:   %13 = fadd fast double %"s.0'de.1", %"s.1'de.1"
; CHECK-NEXT:   %14 = select fast i1 %10, double %13, double %"s.0'de.1"
; CHECK-NEXT:   br i1 %10, label %invertouter, label %incinvertinner

; CHECK: incinvertinner:
; CHECK-NEXT:   %15 = add nsw i64 %"iv1'ac.1", -1
; CHECK-NEXT:   %16 = fadd fast double %"sq'de.1", %12
; CHECK-NEXT:   %17 = add nsw i64 %M, 1
; CHECK-NEXT:   %18 = mul nuw nsw i64 %9, %17
; CHECK-NEXT:   %19 = add nuw nsw i64 %15, %18
; CHECK-NEXT:   %20 = getelementptr inbounds double, ptr %v_cache.0, i64 %19
; CHECK-NEXT:   %21 = load double, ptr %20, align 8, !invariant.group !15
; CHECK-NEXT:   %22 = fmul fast double %16, %21
; CHECK-NEXT:   %23 = fadd fast double %"v'de.1", %22
; CHECK-NEXT:   %24 = fmul fast double %16, %21
; CHECK-NEXT:   %25 = fadd fast double %23, %24
; CHECK-NEXT:   %26 = getelementptr inbounds i64, ptr %off_cache.0, i64 %9
; CHECK-NEXT:   %27 = load i64, ptr %26, align 8, !alias.scope !4, !noalias !7, !invariant.group !9
; CHECK-NEXT:   %"base'ipg_unwrap" = getelementptr double, ptr %"A'", i64 %27
; CHECK-NEXT:   %"stp'ipg_unwrap" = getelementptr double, ptr %"base'ipg_unwrap", i64 %15
; CHECK-NEXT:   store double 0.000000e+00, ptr %"stp'ipg_unwrap", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %"hi'ipg_unwrap" = getelementptr double, ptr %"base'ipg_unwrap", i64 %M
; CHECK-NEXT:   %"hi1'ipg_unwrap" = getelementptr double, ptr %"hi'ipg_unwrap", i64 1
; CHECK-NEXT:   %"ldp'ipg_unwrap" = getelementptr double, ptr %"hi1'ipg_unwrap", i64 %15
; CHECK-NEXT:   %28 = load double, ptr %"ldp'ipg_unwrap", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %29 = fadd fast double %28, %25
; CHECK-NEXT:   store double %29, ptr %"ldp'ipg_unwrap", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   br label %invertinner

; CHECK: invertcleanup:
; CHECK-NEXT:   %off_cache.0 = phi ptr [ undef, %entry ], [ %[[offcache]], %outer ]
; CHECK-NEXT:   %v_cache.0 = phi ptr [ undef, %entry ], [ %[[vcache]], %outer ]
; CHECK-NEXT:   %30 = select fast i1 %or.cond, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %or.cond, label %invertentry, label %invertouter
; CHECK-NEXT: }
