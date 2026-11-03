; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Reverse mode through OpenMP reductions (__kmpc_reduce / __kmpc_reduce_nowait),
; preserving the O(log P) runtime tree reduction rather than serializing with a
; critical section.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 18, i32 0, i32 0, ptr @0 }, align 8
@.gomp_critical_user_.reduction.var = common global [8 x i32] zeroinitializer, align 8

define void @caller(ptr %x, ptr %dx, ptr %s, ptr %ds, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @sumsq, ptr %x, ptr %dx, ptr %s, ptr %ds, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @maxsq, ptr %x, ptr %dx, ptr %s, ptr %ds, i64 %n)
  ret void
}

declare void @__enzyme_autodiff(...)

define internal void @sumsq(ptr %x, ptr %s, i64 %n) {
entry:
  store double 0.000000e+00, ptr %s, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @2, i32 3, ptr @.omp_outlined., i64 %n, ptr %x, ptr %s)
  ret void
}

define internal void @.omp_outlined.(ptr noalias nocapture readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 %n, ptr nocapture readonly %x, ptr nocapture %s) {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %priv = alloca double, align 8
  %red.array = alloca [1 x ptr], align 8
  store double 0.000000e+00, ptr %priv, align 8
  %sub = add i64 %n, -1
  store i64 0, ptr %.omp.lb, align 8
  store i64 %sub, ptr %.omp.ub, align 8
  store i64 1, ptr %.omp.stride, align 8
  store i32 0, ptr %.omp.is_last, align 4
  %tid = load i32, ptr %.global_tid., align 4
  call void @__kmpc_for_static_init_8u(ptr nonnull @1, i32 %tid, i32 34, ptr nonnull %.omp.is_last, ptr nonnull %.omp.lb, ptr nonnull %.omp.ub, ptr nonnull %.omp.stride, i64 1, i64 1)
  %ub = load i64, ptr %.omp.ub, align 8
  %cmp6 = icmp ugt i64 %ub, %sub
  %cond = select i1 %cmp6, i64 %sub, i64 %ub
  %lb = load i64, ptr %.omp.lb, align 8
  %add29 = add i64 %cond, 1
  %cmp730 = icmp ult i64 %lb, %add29
  br i1 %cmp730, label %omp.inner.for.body, label %omp.loop.exit

omp.inner.for.body:
  %iv = phi i64 [ %iv.next, %omp.inner.for.body ], [ %lb, %entry ]
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %iv
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  %acc = load double, ptr %priv, align 8
  %acc.next = fadd double %acc, %sq
  store double %acc.next, ptr %priv, align 8
  %iv.next = add nuw i64 %iv, 1
  %cmp7 = icmp ult i64 %iv.next, %add29
  br i1 %cmp7, label %omp.inner.for.body, label %omp.loop.exit

omp.loop.exit:
  call void @__kmpc_for_static_fini(ptr nonnull @1, i32 %tid)
  store ptr %priv, ptr %red.array, align 8
  %reduce = call i32 @__kmpc_reduce_nowait(ptr nonnull @2, i32 %tid, i32 1, i64 8, ptr nonnull %red.array, ptr nonnull @.omp.reduction.func, ptr nonnull @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %omp.reduction.default [
    i32 1, label %omp.reduction.case1
  ]

omp.reduction.case1:
  %shared = load double, ptr %s, align 8
  %mine = load double, ptr %priv, align 8
  %sum = fadd double %shared, %mine
  store double %sum, ptr %s, align 8
  call void @__kmpc_end_reduce_nowait(ptr nonnull @2, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)
  br label %omp.reduction.default

omp.reduction.default:
  ret void
}

define internal void @.omp.reduction.func(ptr %0, ptr %1) {
  %lhsp = load ptr, ptr %0, align 8
  %rhsp = load ptr, ptr %1, align 8
  %lhs = load double, ptr %lhsp, align 8
  %rhs = load double, ptr %rhsp, align 8
  %add = fadd double %lhs, %rhs
  store double %add, ptr %lhsp, align 8
  ret void
}

define internal void @maxsq(ptr %x, ptr %m, i64 %n) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @2, i32 3, ptr @.omp_outlined.max, i64 %n, ptr %x, ptr %m)
  ret void
}

; Blocking __kmpc_reduce with KMP_IDENT_ATOMIC_REDUCE (0x10) set in @3 and a
; case 2 atomic fallback block, verifying CanonicalizeOMPReductions + reverse
; tree reduction for max.
define internal void @.omp_outlined.max(ptr noalias nocapture readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 %n, ptr nocapture readonly %x, ptr nocapture %m) {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %priv = alloca double, align 8
  %red.array = alloca [1 x ptr], align 8
  store double -1.000000e+30, ptr %priv, align 8
  %sub = add i64 %n, -1
  store i64 0, ptr %.omp.lb, align 8
  store i64 %sub, ptr %.omp.ub, align 8
  store i64 1, ptr %.omp.stride, align 8
  store i32 0, ptr %.omp.is_last, align 4
  %tid = load i32, ptr %.global_tid., align 4
  call void @__kmpc_for_static_init_8u(ptr nonnull @1, i32 %tid, i32 34, ptr nonnull %.omp.is_last, ptr nonnull %.omp.lb, ptr nonnull %.omp.ub, ptr nonnull %.omp.stride, i64 1, i64 1)
  %ub = load i64, ptr %.omp.ub, align 8
  %cmp6 = icmp ugt i64 %ub, %sub
  %cond = select i1 %cmp6, i64 %sub, i64 %ub
  %lb = load i64, ptr %.omp.lb, align 8
  %add29 = add i64 %cond, 1
  %cmp730 = icmp ult i64 %lb, %add29
  br i1 %cmp730, label %omp.inner.for.body, label %omp.loop.exit

omp.inner.for.body:
  %iv = phi i64 [ %iv.next, %omp.inner.for.body ], [ %lb, %entry ]
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %iv
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  %cur = load double, ptr %priv, align 8
  %gt = fcmp ogt double %sq, %cur
  %nxt = select i1 %gt, double %sq, double %cur
  store double %nxt, ptr %priv, align 8
  %iv.next = add nuw i64 %iv, 1
  %cmp7 = icmp ult i64 %iv.next, %add29
  br i1 %cmp7, label %omp.inner.for.body, label %omp.loop.exit

omp.loop.exit:
  call void @__kmpc_for_static_fini(ptr nonnull @1, i32 %tid)
  store ptr %priv, ptr %red.array, align 8
  %reduce = call i32 @__kmpc_reduce(ptr nonnull @3, i32 %tid, i32 1, i64 8, ptr nonnull %red.array, ptr nonnull @.omp.reduction.max, ptr nonnull @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %omp.reduction.default [
    i32 1, label %omp.reduction.case1
    i32 2, label %omp.reduction.case2
  ]

omp.reduction.case1:
  %shared = load double, ptr %m, align 8
  %mine = load double, ptr %priv, align 8
  %lt = fcmp olt double %shared, %mine
  %sel = select i1 %lt, double %mine, double %shared
  store double %sel, ptr %m, align 8
  call void @__kmpc_end_reduce(ptr nonnull @3, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)
  br label %omp.reduction.default

omp.reduction.case2:
  %mine2 = load double, ptr %priv, align 8
  store atomic double %mine2, ptr %m monotonic, align 8
  br label %omp.reduction.default

omp.reduction.default:
  ret void
}

define internal void @.omp.reduction.max(ptr %0, ptr %1) {
  %lhsp = load ptr, ptr %0, align 8
  %rhsp = load ptr, ptr %1, align 8
  %lhs = load double, ptr %lhsp, align 8
  %rhs = load double, ptr %rhsp, align 8
  %cmp = fcmp ogt double %lhs, %rhs
  %max = select i1 %cmp, double %lhs, double %rhs
  store double %max, ptr %lhsp, align 8
  ret void
}

declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)
declare void @__kmpc_for_static_fini(ptr, i32)
declare i32 @__kmpc_reduce(ptr, i32, i32, i64, ptr, ptr, ptr)
declare void @__kmpc_end_reduce(ptr, i32, ptr)
declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr)
declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr)
declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: @.noatomic = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }

; CHECK-LABEL: define internal void @augmented_.omp_outlined..1(
; CHECK-NOT: @__kmpc_critical(
; CHECK: %reduce = call i32 @__kmpc_reduce_nowait(ptr nonnull @2, i32 %tid, i32 1, i64 8, ptr nonnull %{{.+}}, ptr nonnull @.omp.reduction.func, ptr nonnull @.gomp_critical_user_.reduction.var)
; CHECK: omp.reduction.case1:
; CHECK: call void @__kmpc_end_reduce_nowait(ptr nonnull @2, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)

; CHECK-LABEL: define internal void @diffe.omp_outlined.(
; CHECK-NOT: @__kmpc_reduce_nowait(
; CHECK: call void @__kmpc_barrier(ptr @2, i32 %tid)
; CHECK: call void @__kmpc_barrier(ptr @2, i32 %tid)
; CHECK: call void @__kmpc_end_critical(ptr @2, i32 %tid, ptr @__enzyme_omp_reduce_crit_lck)
; CHECK: invertomp.reduction.case1:
; CHECK-NEXT: call void @__kmpc_critical(ptr @2, i32 %tid, ptr @__enzyme_omp_reduce_crit_lck)
; CHECK: atomicrmw fadd ptr %"s'"

; CHECK-LABEL: define internal void @augmented_.omp_outlined.max.{{[0-9]+}}(
; CHECK-NOT: store atomic
; CHECK: %omp_reduce_pre = load double
; CHECK: %reduce = call i32 @__kmpc_reduce_nowait(ptr nonnull @.noatomic, i32 %tid, i32 1, i64 8, ptr nonnull %{{.+}}, ptr nonnull @.omp.reduction.max, ptr nonnull @.gomp_critical_user_.reduction.var)
; CHECK: omp.reduction.case1:
; CHECK: call void @__kmpc_end_reduce_nowait(ptr nonnull @.noatomic, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)
; CHECK: omp.reduction.default:
; CHECK-NEXT: call void @__kmpc_barrier(ptr @.noatomic, i32 %tid)

; CHECK-LABEL: define internal void @diffe.omp_outlined.max(
; CHECK: call i32 @__kmpc_reduce_nowait(ptr @.noatomic, i32 %tid, i32 1, i64 24, ptr %omp_rev_mm_local, ptr @__enzyme_omp_rev_maxmin_w1_max_{{[0-9]+}}, ptr @__enzyme_omp_reduce_rev_lck)
; CHECK: call void @__enzyme_omp_rev_maxmin_w1_max_{{[0-9]+}}(
; CHECK: call void @__kmpc_end_reduce_nowait(ptr @.noatomic, i32 %tid, ptr @__enzyme_omp_reduce_rev_lck)
