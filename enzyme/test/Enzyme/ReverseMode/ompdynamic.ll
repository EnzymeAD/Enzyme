; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Reverse mode through a worksharing loop with a dynamic schedule. Enzyme runs
; it with the static schedule: the first __kmpc_dispatch_next becomes
; __kmpc_for_static_init on the bounds given to __kmpc_dispatch_init, and the
; loop over chunks goes away.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

define void @caller(ptr %x, ptr %dx, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @square, ptr %x, ptr %dx, i64 %n)
  ret void
}

declare void @__enzyme_autodiff(...)

define internal void @square(ptr %x, i64 %n) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @2, i32 2, ptr @.omp_outlined., i64 %n, ptr %x)
  ret void
}

define internal void @.omp_outlined.(ptr noalias nocapture readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 %n, ptr nocapture %x) {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %sub = add i64 %n, -1
  %tid = load i32, ptr %.global_tid., align 4
  call void @__kmpc_dispatch_init_8u(ptr nonnull @1, i32 %tid, i32 1073741859, i64 0, i64 %sub, i64 1, i64 1)
  br label %dispatch.cond

dispatch.cond:
  %more = call i32 @__kmpc_dispatch_next_8u(ptr nonnull @1, i32 %tid, ptr nonnull %.omp.is_last, ptr nonnull %.omp.lb, ptr nonnull %.omp.ub, ptr nonnull %.omp.stride)
  %done = icmp eq i32 %more, 0
  br i1 %done, label %exit, label %chunk

chunk:
  %lb = load i64, ptr %.omp.lb, align 8
  %ub = load i64, ptr %.omp.ub, align 8
  %ub1 = add i64 %ub, 1
  %nonempty = icmp ult i64 %lb, %ub1
  br i1 %nonempty, label %body, label %dispatch.cond

body:
  %iv = phi i64 [ %iv.next, %body ], [ %lb, %chunk ]
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %iv
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %arrayidx, align 8
  %iv.next = add nuw i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, %ub1
  br i1 %cmp, label %body, label %dispatch.cond

exit:
  ret void
}

declare void @__kmpc_dispatch_init_8u(ptr, i32, i32, i64, i64, i64, i64)
declare i32 @__kmpc_dispatch_next_8u(ptr, i32, ptr, ptr, ptr, ptr)
declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @augmented_.omp_outlined.(
; CHECK-NOT: @__kmpc_dispatch_
; CHECK:   call void @__kmpc_for_static_init_8u(ptr @1, i32 %tid, i32 34, ptr %.omp.is_last, ptr {{.*}}%.omp.lb_smpl, ptr {{.*}}%.omp.ub_smpl, ptr {{.*}}%.omp.stride_smpl, i64 1, i64 1)
; CHECK-NEXT:   %[[ub:.+]] = load i64, ptr %.omp.ub_smpl
; CHECK-NEXT:   %[[lb:.+]] = load i64, ptr %.omp.lb_smpl
; CHECK: %[[some:.+]] = icmp ule i64 %[[lb]], %[[ub]]
; CHECK-NEXT:   %[[r:.+]] = zext i1 %[[some]] to i32
; CHECK-NEXT:   %done = icmp eq i32 %[[r]], 0
; CHECK-NEXT:   br i1 %done, label %exit, label %chunk
; CHECK: ret void

; CHECK: define internal void @diffe.omp_outlined.(
; CHECK-NOT: @__kmpc_dispatch_
; CHECK:   call void @__kmpc_for_static_init_8u(ptr @1, i32 %tid, i32 34,
; CHECK: invertentry:
; CHECK-NEXT:   call void @__kmpc_for_static_fini(ptr @1, i32 %tid)
; CHECK: atomicrmw fadd ptr %"arrayidx'ipg_unwrap"
