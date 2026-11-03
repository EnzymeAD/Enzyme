; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Forward mode directly through an OpenMP tree reduction (__kmpc_reduce_nowait).
; Verifies that Enzyme bundles the primal and shadow reduction arrays and
; synthesizes a forward-mode combiner wrapper rather than serializing with a
; critical section.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8
@.gomp_critical_user_.reduction.var = common global [8 x i32] zeroinitializer, align 8

define void @caller(ptr %tid_p, ptr %btid_p, i64 %n, ptr %x, ptr %dx, ptr %s, ptr %ds) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @outlined_sum_prod, metadata !"enzyme_const", ptr %tid_p, metadata !"enzyme_const", ptr %btid_p, metadata !"enzyme_const", i64 %n, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %s, ptr %ds)
  ret void
}

declare void @__enzyme_fwddiff(...)

define internal void @outlined_sum_prod(ptr noalias nocapture readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 %n, ptr nocapture readonly %x, ptr nocapture %s) {
entry:
  %priv = alloca double, align 8
  %red.array = alloca [1 x ptr], align 8
  %tid = load i32, ptr %.global_tid., align 4
  %idx = zext i32 %tid to i64
  %gep = getelementptr inbounds double, ptr %x, i64 %idx
  %v = load double, ptr %gep, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %priv, align 8
  store ptr %priv, ptr %red.array, align 8
  %reduce = call i32 @__kmpc_reduce_nowait(ptr nonnull @2, i32 %tid, i32 1, i64 8, ptr nonnull %red.array, ptr nonnull @.omp.reduction.mul, ptr nonnull @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %omp.reduction.default [
    i32 1, label %omp.reduction.case1
  ]

omp.reduction.case1:
  %shared = load double, ptr %s, align 8
  %mine = load double, ptr %priv, align 8
  %prod = fmul double %shared, %mine
  store double %prod, ptr %s, align 8
  call void @__kmpc_end_reduce_nowait(ptr nonnull @2, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)
  br label %omp.reduction.default

omp.reduction.default:
  ret void
}

define internal void @.omp.reduction.mul(ptr %0, ptr %1) {
  %lhsp = load ptr, ptr %0, align 8
  %rhsp = load ptr, ptr %1, align 8
  %lhs = load double, ptr %lhsp, align 8
  %rhs = load double, ptr %rhsp, align 8
  %mul = fmul double %lhs, %rhs
  store double %mul, ptr %lhsp, align 8
  ret void
}

declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr)
declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr)

; CHECK-LABEL: define internal void @fwddiffeoutlined_sum_prod(
; CHECK-NOT: @__kmpc_critical(
; CHECK: %omp_fwd_reduce_bundle = alloca [2 x ptr]
; CHECK: call i32 @__kmpc_reduce_nowait(ptr nonnull @2, i32 %tid, i32 2, i64 16, ptr nonnull %omp_fwd_reduce_bundle, ptr nonnull @__enzyme_omp_fwd_reduce_fwddiffe.omp.reduction.mul_w1, ptr nonnull @.gomp_critical_user_.reduction.var)
; CHECK: omp.reduction.case1:
; CHECK: call void @__kmpc_end_reduce_nowait(ptr nonnull @2, i32 %tid, ptr nonnull @.gomp_critical_user_.reduction.var)

; CHECK-LABEL: define internal void @__enzyme_omp_fwd_reduce_fwddiffe.omp.reduction.mul_w1(
; CHECK: call void @fwddiffe.omp.reduction.mul(
