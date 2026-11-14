; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.experimental.noalias.scope.decl(metadata)

@.fmt1 = private unnamed_addr constant [29 x i8] c"restrict: %g %g %g %g %g %g\0A\00"
@.fmt2 = private unnamed_addr constant [26 x i8] c"hoist: %g %g %g %g %g %g\0A\00"

; 1. Inlined restrict helper inside a loop with indirect indices:
; !alias.scope / !noalias are declared inside the loop via
; llvm.experimental.noalias.scope.decl, so they hold only within a single
; iteration and must NOT be trusted across iterations.
define void @inlined_restrict_loop(ptr nocapture %A, ptr nocapture readonly %ridx, ptr nocapture readonly %widx, ptr noalias nocapture writeonly %out, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi double [ 0.0, %entry ], [ %s.next, %loop ]
  %pr = getelementptr inbounds i32, ptr %ridx, i64 %i
  %ri32 = load i32, ptr %pr, align 4
  %ri = sext i32 %ri32 to i64
  %pa = getelementptr inbounds double, ptr %A, i64 %ri
  %pw = getelementptr inbounds i32, ptr %widx, i64 %i
  %wi32 = load i32, ptr %pw, align 4
  %wi = sext i32 %wi32 to i64
  %pb = getelementptr inbounds double, ptr %A, i64 %wi
  call void @llvm.experimental.noalias.scope.decl(metadata !9)
  call void @llvm.experimental.noalias.scope.decl(metadata !12)
  %v = load double, ptr %pa, align 8, !alias.scope !9, !noalias !12
  store double 9.0, ptr %pb, align 8, !alias.scope !12, !noalias !9
  %sq = fmul double %v, %v
  %s.next = fadd double %s, %sq
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %N
  br i1 %done, label %exit, label %loop

exit:
  store double %s.next, ptr %out, align 8
  ret void
}

; 2. Cross-iteration write between preheader and load inside loop:
; Loop 1 updates A[i+1] += 5 then reads A[i]; Loop 2 overwrites A[0..N-1].
; Enzyme must NOT hoist a preheader memcpy of A[0..N-1] past the cross-iteration
; store to A[i+1].
define void @memcpy_hoist_guard(ptr nocapture %A, ptr noalias nocapture writeonly %out, i64 %N) {
entry:
  br label %loop1

loop1:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop1 ]
  %s = phi double [ 0.0, %entry ], [ %s.next, %loop1 ]
  %ip1 = add nuw nsw i64 %i, 1
  %pp1 = getelementptr inbounds double, ptr %A, i64 %ip1
  %old = load double, ptr %pp1, align 8
  %upd = fadd double %old, 5.0
  store double %upd, ptr %pp1, align 8
  %pi = getelementptr inbounds double, ptr %A, i64 %i
  %v = load double, ptr %pi, align 8
  %sq = fmul double %v, %v
  %s.next = fadd double %s, %sq
  %i.next = add nuw nsw i64 %i, 1
  %done1 = icmp eq i64 %i.next, %N
  br i1 %done1, label %loop2, label %loop1

loop2:
  %j = phi i64 [ 0, %loop1 ], [ %j.next, %loop2 ]
  %pj = getelementptr inbounds double, ptr %A, i64 %j
  store double 99.0, ptr %pj, align 8
  %j.next = add nuw nsw i64 %j, 1
  %done2 = icmp eq i64 %j.next, %N
  br i1 %done2, label %exit, label %loop2

exit:
  store double %s.next, ptr %out, align 8
  ret void
}

define i32 @main() {
entry:
  ; --- Test 1: inlined_restrict_loop ---
  %A1 = alloca [6 x double], align 8
  %dA1 = alloca [6 x double], align 8
  %ridx = alloca [4 x i32], align 4
  %widx = alloca [4 x i32], align 4
  %out1 = alloca double, align 8
  %dout1 = alloca double, align 8
  br label %init1
init1:
  %k1 = phi i64 [ 0, %entry ], [ %k1.n, %init1 ]
  %k1p1 = add i64 %k1, 1
  %k1f = sitofp i64 %k1p1 to double
  %pA1 = getelementptr double, ptr %A1, i64 %k1
  store double %k1f, ptr %pA1, align 8
  %k1.n = add i64 %k1, 1
  %c1 = icmp eq i64 %k1.n, 6
  br i1 %c1, label %run1, label %init1
run1:
  ; ridx = {1, 2, 3, 4}, widx = {0, 1, 2, 3}
  %r0 = getelementptr i32, ptr %ridx, i64 0
  %r1 = getelementptr i32, ptr %ridx, i64 1
  %r2 = getelementptr i32, ptr %ridx, i64 2
  %r3 = getelementptr i32, ptr %ridx, i64 3
  store i32 1, ptr %r0
  store i32 2, ptr %r1
  store i32 3, ptr %r2
  store i32 4, ptr %r3
  %w0 = getelementptr i32, ptr %widx, i64 0
  %w1 = getelementptr i32, ptr %widx, i64 1
  %w2 = getelementptr i32, ptr %widx, i64 2
  %w3 = getelementptr i32, ptr %widx, i64 3
  store i32 0, ptr %w0
  store i32 1, ptr %w1
  store i32 2, ptr %w2
  store i32 3, ptr %w3
  call void @llvm.memset.p0.i64(ptr %dA1, i8 0, i64 48, i1 false)
  store double 0.0, ptr %out1
  store double 1.0, ptr %dout1
  call void (...) @__enzyme_autodiff(ptr @inlined_restrict_loop, ptr %A1, ptr %dA1, metadata !"enzyme_const", ptr %ridx, metadata !"enzyme_const", ptr %widx, ptr %out1, ptr %dout1, i64 4)
  %d1p0 = getelementptr double, ptr %dA1, i64 0
  %d1p1 = getelementptr double, ptr %dA1, i64 1
  %d1p2 = getelementptr double, ptr %dA1, i64 2
  %d1p3 = getelementptr double, ptr %dA1, i64 3
  %d1p4 = getelementptr double, ptr %dA1, i64 4
  %d1p5 = getelementptr double, ptr %dA1, i64 5
  %d1v0 = load double, ptr %d1p0
  %d1v1 = load double, ptr %d1p1
  %d1v2 = load double, ptr %d1p2
  %d1v3 = load double, ptr %d1p3
  %d1v4 = load double, ptr %d1p4
  %d1v5 = load double, ptr %d1p5
  call i32 (ptr, ...) @printf(ptr @.fmt1, double %d1v0, double %d1v1, double %d1v2, double %d1v3, double %d1v4, double %d1v5)

  ; --- Test 2: memcpy_hoist_guard ---
  %A2 = alloca [6 x double], align 8
  %dA2 = alloca [6 x double], align 8
  %out2 = alloca double, align 8
  %dout2 = alloca double, align 8
  br label %init2
init2:
  %k2 = phi i64 [ 0, %run1 ], [ %k2.n, %init2 ]
  %k2p1 = add i64 %k2, 1
  %k2f = sitofp i64 %k2p1 to double
  %pA2 = getelementptr double, ptr %A2, i64 %k2
  store double %k2f, ptr %pA2, align 8
  %k2.n = add i64 %k2, 1
  %c2 = icmp eq i64 %k2.n, 6
  br i1 %c2, label %run2, label %init2
run2:
  call void @llvm.memset.p0.i64(ptr %dA2, i8 0, i64 48, i1 false)
  store double 0.0, ptr %out2
  store double 1.0, ptr %dout2
  call void (...) @__enzyme_autodiff(ptr @memcpy_hoist_guard, ptr %A2, ptr %dA2, ptr %out2, ptr %dout2, i64 4)
  %d2p0 = getelementptr double, ptr %dA2, i64 0
  %d2p1 = getelementptr double, ptr %dA2, i64 1
  %d2p2 = getelementptr double, ptr %dA2, i64 2
  %d2p3 = getelementptr double, ptr %dA2, i64 3
  %d2p4 = getelementptr double, ptr %dA2, i64 4
  %d2p5 = getelementptr double, ptr %dA2, i64 5
  %d2v0 = load double, ptr %d2p0
  %d2v1 = load double, ptr %d2p1
  %d2v2 = load double, ptr %d2p2
  %d2v3 = load double, ptr %d2p3
  %d2v4 = load double, ptr %d2p4
  %d2v5 = load double, ptr %d2p5
  call i32 (ptr, ...) @printf(ptr @.fmt2, double %d2v0, double %d2v1, double %d2v2, double %d2v3, double %d2v4, double %d2v5)

  ret i32 0
}

!9 = !{!10}
!10 = distinct !{!10, !11, !"step: argument 0"}
!11 = distinct !{!11, !"step"}
!12 = !{!13}
!13 = distinct !{!13, !11, !"step: argument 1"}

; Verify inlined_restrict_loop caches %v rather than trusting the loop-local noalias scopes.
; CHECK-LABEL: define internal void @diffeinlined_restrict_loop(
; CHECK: @malloc(

; Verify memcpy_hoist_guard does NOT hoist a preheader llvm.memcpy of %A before loop1.
; CHECK-LABEL: define internal void @diffememcpy_hoist_guard(
; CHECK-NEXT: entry:
; CHECK-NOT: llvm.memcpy
; CHECK: br label %loop1

; EVAL: restrict: 0 4 6 8 10 0
; EVAL-NEXT: hoist: 2 14 16 18 0 0
