; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

@.fmt1 = private unnamed_addr constant [21 x i8] c"nested: %g %g %g %g\0A\00"
@.fmt2 = private unnamed_addr constant [25 x i8] c"shift_left: %g %g %g %g\0A\00"
@.fmt3 = private unnamed_addr constant [21 x i8] c"unknown_step: %g %g\0A\00"
@.fmt4 = private unnamed_addr constant [27 x i8] c"nested_lex: %g %g / %g %g\0A\00"
@.fmt5 = private unnamed_addr constant [20 x i8] c"lcssa: %g %g %g %g\0A\00"
@.fmt6 = private unnamed_addr constant [26 x i8] c"shift_right: %g %g %g %g\0A\00"

; 1. Issue #3448: nested loop with shared outer i*K and constant inner offset.
; BasicAA cancels shared i*K intra-iteration and returns NoAlias for A[i*K+j+4] vs A[i*K+j],
; even though iteration i+1 writes A[(i+1)*K+j] == A[i*K+j+4] when K == 4.
define void @nested_offset(ptr nocapture %A, i64 %N, i64 %K) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %base = mul i64 %i, %K
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = add i64 %base, %j
  %idx4 = add i64 %idx, 4
  %pl = getelementptr inbounds double, ptr %A, i64 %idx4
  %ps = getelementptr inbounds double, ptr %A, i64 %idx
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %ps, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %N
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}

; 2. 1D left-shift: A[i] = A[i+1]^2. BasicAA returns NoAlias(A[i+1], A[i]) intra-iteration,
; but iteration i+1 overwrites A[i+1] which was read in iteration i.
define void @shift_left(ptr nocapture %A, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %ip1 = add nuw nsw i64 %i, 1
  %pl = getelementptr inbounds double, ptr %A, i64 %ip1
  %ps = getelementptr inbounds double, ptr %A, i64 %i
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %ps, align 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %N
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 3. Unknown-sign step K in overwritesToMemoryReadByLoop:
; load A[4 + i*K], store A[i*K]. Because sign(K) is unknown, StoreEnd(w) <= LoadStart(r)
; does NOT hold for w > r when K > 0.
define void @unknown_step(ptr nocapture %A, i64 %K, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %idx2 = phi i64 [ 4, %entry ], [ %idx2.next, %loop ]
  %idx1 = mul i64 %i, %K
  %ps = getelementptr inbounds double, ptr %A, i64 %idx1
  %pl = getelementptr inbounds double, ptr %A, i64 %idx2
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %ps, align 8
  %idx2.next = add i64 %idx2, %K
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %N
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 4a. Nested common loops lexicographic reset (positive steps):
; A[2*i + 3*j] read, A[2*i + 3*j + 1] written -> inner loop resets on outer increment,
; so (i=1, j=0) writes A[3] which was read at (i=0, j=1) (2*0+3*1 == 3).
define void @nested_lex_pos(ptr nocapture %A, ptr noalias nocapture writeonly %out) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %s.out = phi double [ 0.0, %entry ], [ %s.in.next, %outer.latch ]
  %i2 = mul nuw nsw i64 %i, 2
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %s.in = phi double [ %s.out, %outer ], [ %s.in.next, %inner ]
  %j3 = mul nuw nsw i64 %j, 3
  %idx = add nuw nsw i64 %i2, %j3
  %idxp1 = add nuw nsw i64 %idx, 1
  %pl = getelementptr inbounds double, ptr %A, i64 %idx
  %ps = getelementptr inbounds double, ptr %A, i64 %idxp1
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  %s.in.next = fadd double %s.in, %sq
  store double 99.0, ptr %ps, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 2
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 2
  br i1 %i.done, label %exit, label %outer

exit:
  store double %s.in.next, ptr %out, align 8
  ret void
}

; 4b. Nested common loops lexicographic reset (negative steps):
; A[20 - 2*i - 3*j + 1] read, A[20 - 2*i - 3*j] written ->
; (i=0, j=1) reads A[18], (i=1, j=0) writes A[18].
define void @nested_lex_neg(ptr nocapture %A, ptr noalias nocapture writeonly %out) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %s.out = phi double [ 0.0, %entry ], [ %s.in.next, %outer.latch ]
  %i2 = mul nuw nsw i64 %i, 2
  %base = sub nsw i64 20, %i2
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %s.in = phi double [ %s.out, %outer ], [ %s.in.next, %inner ]
  %j3 = mul nuw nsw i64 %j, 3
  %idx = sub nsw i64 %base, %j3
  %idxp1 = add nsw i64 %idx, 1
  %pl = getelementptr inbounds double, ptr %A, i64 %idxp1
  %ps = getelementptr inbounds double, ptr %A, i64 %idx
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  %s.in.next = fadd double %s.in, %sq
  store double 99.0, ptr %ps, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 2
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 2
  br i1 %i.done, label %exit, label %outer

exit:
  store double %s.in.next, ptr %out, align 8
  ret void
}

; 5. Load inside loop vs store AFTER loop via LCSSA PHI:
; SE.getSCEV(%pm2) returns an in-loop AddRec unless evaluated at the writer's scope
; with SE.getSCEVAtScope, and BasicAA treats %p and %pm2 as same-iteration NoAlias.
define void @lcssa_exit(ptr nocapture %A, ptr noalias nocapture writeonly %out, i64 %N) {
entry:
  %cmp0 = icmp sgt i64 %N, 2
  br i1 %cmp0, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %p = phi ptr [ %A, %entry ], [ %pnext, %loop ]
  %s = phi double [ 0.0, %entry ], [ %snext, %loop ]
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %snext = fadd double %s, %sq
  %pnext = getelementptr inbounds double, ptr %p, i64 1
  %inc = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %inc, %N
  br i1 %cond, label %loop, label %after

after:
  %p_lcssa = phi ptr [ %pnext, %loop ]
  %s_lcssa = phi double [ %snext, %loop ]
  %pm2 = getelementptr inbounds double, ptr %p_lcssa, i64 -2
  store double 99.0, ptr %pm2, align 8
  br label %exit

exit:
  %sfin = phi double [ 0.0, %entry ], [ %s_lcssa, %after ]
  store double %sfin, ptr %out, align 8
  ret void
}

; 6. Positive precision test: 1D right-shift A[i+1] = A[i]^2.
; Iteration r reads A[r], iteration w >= r writes A[w+1] > A[r], so A[r] is
; NEVER overwritten and %v should NOT be cached (no malloc).
define void @shift_right_no_cache(ptr nocapture %A, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %ip1 = add nuw nsw i64 %i, 1
  %pl = getelementptr inbounds double, ptr %A, i64 %i
  %ps = getelementptr inbounds double, ptr %A, i64 %ip1
  %v = load double, ptr %pl, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %ps, align 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %N
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
entry:
  ; --- Test 1: nested_offset (N=2, K=4, A[k]=k+1, dA[k]=1) ---
  %A1 = alloca [16 x double], align 8
  %dA1 = alloca [16 x double], align 8
  br label %init1
init1:
  %k1 = phi i64 [ 0, %entry ], [ %k1.n, %init1 ]
  %k1p1 = add i64 %k1, 1
  %k1f = sitofp i64 %k1p1 to double
  %pA1 = getelementptr double, ptr %A1, i64 %k1
  %pdA1 = getelementptr double, ptr %dA1, i64 %k1
  store double %k1f, ptr %pA1, align 8
  store double 1.0, ptr %pdA1, align 8
  %k1.n = add i64 %k1, 1
  %c1 = icmp eq i64 %k1.n, 16
  br i1 %c1, label %run1, label %init1
run1:
  call void (...) @__enzyme_autodiff(ptr @nested_offset, ptr %A1, ptr %dA1, i64 2, i64 4)
  %r1p4 = getelementptr double, ptr %dA1, i64 4
  %r1p5 = getelementptr double, ptr %dA1, i64 5
  %r1p6 = getelementptr double, ptr %dA1, i64 6
  %r1p7 = getelementptr double, ptr %dA1, i64 7
  %r1v4 = load double, ptr %r1p4
  %r1v5 = load double, ptr %r1p5
  %r1v6 = load double, ptr %r1p6
  %r1v7 = load double, ptr %r1p7
  call i32 (ptr, ...) @printf(ptr @.fmt1, double %r1v4, double %r1v5, double %r1v6, double %r1v7)

  ; --- Test 2: shift_left (N=3, A={1,2,3,4,5}, dA={1,1,1,1,1}) ---
  %A2 = alloca [5 x double], align 8
  %dA2 = alloca [5 x double], align 8
  br label %init2
init2:
  %k2 = phi i64 [ 0, %run1 ], [ %k2.n, %init2 ]
  %k2p1 = add i64 %k2, 1
  %k2f = sitofp i64 %k2p1 to double
  %pA2 = getelementptr double, ptr %A2, i64 %k2
  %pdA2 = getelementptr double, ptr %dA2, i64 %k2
  store double %k2f, ptr %pA2, align 8
  store double 1.0, ptr %pdA2, align 8
  %k2.n = add i64 %k2, 1
  %c2 = icmp eq i64 %k2.n, 5
  br i1 %c2, label %run2, label %init2
run2:
  call void (...) @__enzyme_autodiff(ptr @shift_left, ptr %A2, ptr %dA2, i64 3)
  %r2p0 = getelementptr double, ptr %dA2, i64 0
  %r2p1 = getelementptr double, ptr %dA2, i64 1
  %r2p2 = getelementptr double, ptr %dA2, i64 2
  %r2p3 = getelementptr double, ptr %dA2, i64 3
  %r2v0 = load double, ptr %r2p0
  %r2v1 = load double, ptr %r2p1
  %r2v2 = load double, ptr %r2p2
  %r2v3 = load double, ptr %r2p3
  call i32 (ptr, ...) @printf(ptr @.fmt2, double %r2v0, double %r2v1, double %r2v2, double %r2v3)

  ; --- Test 3: unknown_step (K=4, N=3, A[k]=k+1, dA[k]=1) ---
  %A3 = alloca [16 x double], align 8
  %dA3 = alloca [16 x double], align 8
  br label %init3
init3:
  %k3 = phi i64 [ 0, %run2 ], [ %k3.n, %init3 ]
  %k3p1 = add i64 %k3, 1
  %k3f = sitofp i64 %k3p1 to double
  %pA3 = getelementptr double, ptr %A3, i64 %k3
  %pdA3 = getelementptr double, ptr %dA3, i64 %k3
  store double %k3f, ptr %pA3, align 8
  store double 1.0, ptr %pdA3, align 8
  %k3.n = add i64 %k3, 1
  %c3 = icmp eq i64 %k3.n, 16
  br i1 %c3, label %run3, label %init3
run3:
  call void (...) @__enzyme_autodiff(ptr @unknown_step, ptr %A3, ptr %dA3, i64 4, i64 3)
  %r3p4 = getelementptr double, ptr %dA3, i64 4
  %r3p8 = getelementptr double, ptr %dA3, i64 8
  %r3v4 = load double, ptr %r3p4
  %r3v8 = load double, ptr %r3p8
  call i32 (ptr, ...) @printf(ptr @.fmt3, double %r3v4, double %r3v8)

  ; --- Test 4: nested_lex_pos & nested_lex_neg ---
  %A4a = alloca [24 x double], align 8
  %dA4a = alloca [24 x double], align 8
  %A4b = alloca [24 x double], align 8
  %dA4b = alloca [24 x double], align 8
  %out4 = alloca double, align 8
  %dout4 = alloca double, align 8
  br label %init4
init4:
  %k4 = phi i64 [ 0, %run3 ], [ %k4.n, %init4 ]
  %k4p1 = add i64 %k4, 1
  %k4f = sitofp i64 %k4p1 to double
  %pA4a = getelementptr double, ptr %A4a, i64 %k4
  %pA4b = getelementptr double, ptr %A4b, i64 %k4
  store double %k4f, ptr %pA4a, align 8
  store double %k4f, ptr %pA4b, align 8
  %k4.n = add i64 %k4, 1
  %c4 = icmp eq i64 %k4.n, 24
  br i1 %c4, label %run4, label %init4
run4:
  call void @llvm.memset.p0.i64(ptr %dA4a, i8 0, i64 192, i1 false)
  call void @llvm.memset.p0.i64(ptr %dA4b, i8 0, i64 192, i1 false)
  store double 0.0, ptr %out4
  store double 1.0, ptr %dout4
  call void (...) @__enzyme_autodiff(ptr @nested_lex_pos, ptr %A4a, ptr %dA4a, ptr %out4, ptr %dout4)
  store double 0.0, ptr %out4
  store double 1.0, ptr %dout4
  call void (...) @__enzyme_autodiff(ptr @nested_lex_neg, ptr %A4b, ptr %dA4b, ptr %out4, ptr %dout4)
  %r4ap3 = getelementptr double, ptr %dA4a, i64 3
  %r4ap5 = getelementptr double, ptr %dA4a, i64 5
  %r4bp18 = getelementptr double, ptr %dA4b, i64 18
  %r4bp16 = getelementptr double, ptr %dA4b, i64 16
  %r4av3 = load double, ptr %r4ap3
  %r4av5 = load double, ptr %r4ap5
  %r4bv18 = load double, ptr %r4bp18
  %r4bv16 = load double, ptr %r4bp16
  call i32 (ptr, ...) @printf(ptr @.fmt4, double %r4av3, double %r4av5, double %r4bv18, double %r4bv16)

  ; --- Test 5: lcssa_exit (N=4, A={1,2,3,4,5}) ---
  %A5 = alloca [5 x double], align 8
  %dA5 = alloca [5 x double], align 8
  %out5 = alloca double, align 8
  %dout5 = alloca double, align 8
  br label %init5
init5:
  %k5 = phi i64 [ 0, %run4 ], [ %k5.n, %init5 ]
  %k5p1 = add i64 %k5, 1
  %k5f = sitofp i64 %k5p1 to double
  %pA5 = getelementptr double, ptr %A5, i64 %k5
  store double %k5f, ptr %pA5, align 8
  %k5.n = add i64 %k5, 1
  %c5 = icmp eq i64 %k5.n, 5
  br i1 %c5, label %run5, label %init5
run5:
  call void @llvm.memset.p0.i64(ptr %dA5, i8 0, i64 40, i1 false)
  store double 0.0, ptr %out5
  store double 1.0, ptr %dout5
  call void (...) @__enzyme_autodiff(ptr @lcssa_exit, ptr %A5, ptr %dA5, ptr %out5, ptr %dout5, i64 4)
  %r5p0 = getelementptr double, ptr %dA5, i64 0
  %r5p1 = getelementptr double, ptr %dA5, i64 1
  %r5p2 = getelementptr double, ptr %dA5, i64 2
  %r5p3 = getelementptr double, ptr %dA5, i64 3
  %r5v0 = load double, ptr %r5p0
  %r5v1 = load double, ptr %r5p1
  %r5v2 = load double, ptr %r5p2
  %r5v3 = load double, ptr %r5p3
  call i32 (ptr, ...) @printf(ptr @.fmt5, double %r5v0, double %r5v1, double %r5v2, double %r5v3)

  ; --- Test 6: shift_right_no_cache (N=3, A={2,1,1,1}, dA={0,0,0,1}) ---
  ; A[1]=A[0]^2=4, A[2]=A[1]^2=16, A[3]=A[2]^2=256 => d(A[3])/d(A[0]) = 8*2^7 = 1024
  %A6 = alloca [4 x double], align 8
  %dA6 = alloca [4 x double], align 8
  store double 2.0, ptr %A6
  %a6p1 = getelementptr double, ptr %A6, i64 1
  %a6p2 = getelementptr double, ptr %A6, i64 2
  %a6p3 = getelementptr double, ptr %A6, i64 3
  store double 1.0, ptr %a6p1
  store double 1.0, ptr %a6p2
  store double 1.0, ptr %a6p3
  call void @llvm.memset.p0.i64(ptr %dA6, i8 0, i64 32, i1 false)
  %da6p3 = getelementptr double, ptr %dA6, i64 3
  store double 1.0, ptr %da6p3
  call void (...) @__enzyme_autodiff(ptr @shift_right_no_cache, ptr %A6, ptr %dA6, i64 3)
  %r6p0 = getelementptr double, ptr %dA6, i64 0
  %r6p1 = getelementptr double, ptr %dA6, i64 1
  %r6p2 = getelementptr double, ptr %dA6, i64 2
  %r6p3 = getelementptr double, ptr %dA6, i64 3
  %r6v0 = load double, ptr %r6p0
  %r6v1 = load double, ptr %r6p1
  %r6v2 = load double, ptr %r6p2
  %r6v3 = load double, ptr %r6p3
  call i32 (ptr, ...) @printf(ptr @.fmt6, double %r6v0, double %r6v1, double %r6v2, double %r6v3)

  ret i32 0
}

; Verify that shift_right_no_cache still proves no overwrite via SCEV and does not allocate a cache.
; CHECK-LABEL: define internal void @diffeshift_right_no_cache(
; CHECK-NOT: @malloc(
; CHECK: ret void

; EVAL: nested: 10 12 14 16
; EVAL-NEXT: shift_left: 0 4 6 9
; EVAL-NEXT: unknown_step: 10 18
; EVAL-NEXT: nested_lex: 8 12 / 38 34
; EVAL-NEXT: lcssa: 2 4 6 8
; EVAL-NEXT: shift_right: 1024 0 0 0
