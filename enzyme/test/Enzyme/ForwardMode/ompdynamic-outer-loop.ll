; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A worksharing loop with a dynamic schedule inside a serial loop of the
; parallel region, as flang emits for
;   !$OMP PARALLEL
;   DO j = ...
;     !$OMP DO SCHEDULE(dynamic,1)
; Optimized, __kmpc_dispatch_init and the first __kmpc_dispatch_next share the
; header of the serial loop, whose latch is dominated by it but enters it
; through the __kmpc_dispatch_init of the next iteration. The static_init
; replacing __kmpc_dispatch_next belongs where it was, after
; __kmpc_dispatch_init and its bounds, and not in the preheader of the serial
; loop.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

define void @caller(ptr %x, ptr %dx, i64 %n, i64 %m) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @square, ptr %x, ptr %dx, i64 %n, i64 %m)
  ret void
}

declare void @__enzyme_fwddiff(...)

define internal void @square(ptr %x, i64 %n, i64 %m) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @2, i32 3, ptr @.omp_outlined., i64 %n, i64 %m, ptr %x)
  ret void
}

define internal void @.omp_outlined.(ptr noalias nocapture readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 %n, i64 %m, ptr nocapture %x) {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %tid0 = load i32, ptr %.global_tid., align 4
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %tid = phi i32 [ %tid0, %entry ], [ %tid.next, %outer.latch ]
  %jn = add i64 %j, %n
  %sub = add i64 %jn, -1
  call void @__kmpc_dispatch_init_8u(ptr nonnull @1, i32 %tid, i32 1073741859, i64 0, i64 %sub, i64 1, i64 1)
  %more = call i32 @__kmpc_dispatch_next_8u(ptr nonnull @1, i32 %tid, ptr nonnull %.omp.is_last, ptr nonnull %.omp.lb, ptr nonnull %.omp.ub, ptr nonnull %.omp.stride)
  %done = icmp eq i32 %more, 0
  br i1 %done, label %outer.latch, label %chunk

chunk:
  %lb = load i64, ptr %.omp.lb, align 8
  %ub = load i64, ptr %.omp.ub, align 8
  %ub1 = add i64 %ub, 1
  %nonempty = icmp ult i64 %lb, %ub1
  br i1 %nonempty, label %body, label %dispatch.next

body:
  %iv = phi i64 [ %iv.next, %body ], [ %lb, %chunk ]
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %iv
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %arrayidx, align 8
  %iv.next = add nuw i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, %ub1
  br i1 %cmp, label %body, label %dispatch.next

dispatch.next:
  %more2 = call i32 @__kmpc_dispatch_next_8u(ptr nonnull @1, i32 %tid, ptr nonnull %.omp.is_last, ptr nonnull %.omp.lb, ptr nonnull %.omp.ub, ptr nonnull %.omp.stride)
  %done2 = icmp eq i32 %more2, 0
  br i1 %done2, label %outer.latch, label %chunk

outer.latch:
  call void @__kmpc_barrier(ptr nonnull @2, i32 %tid)
  %tid.next = call i32 @__kmpc_global_thread_num(ptr nonnull @2)
  %j.next = add nuw i64 %j, 1
  %outer.cmp = icmp ult i64 %j.next, %m
  br i1 %outer.cmp, label %outer, label %exit

exit:
  ret void
}

declare void @__kmpc_dispatch_init_8u(ptr, i32, i32, i64, i64, i64, i64)
declare i32 @__kmpc_dispatch_next_8u(ptr, i32, ptr, ptr, ptr, ptr)
declare void @__kmpc_barrier(ptr, i32)
declare i32 @__kmpc_global_thread_num(ptr)
declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @fwddiffe.omp_outlined.(
; CHECK-NOT: @__kmpc_dispatch_
; CHECK: outer:
; CHECK:   %tid = phi i32
; CHECK:   %sub = add i64 %jn, -1
; CHECK:   store i64 %sub, ptr %.omp.ub
; CHECK:   call void @__kmpc_for_static_init_8u(ptr @1, i32 %tid, i32 34, ptr %.omp.is_last, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i64 1, i64 1)
; CHECK-NOT: @__kmpc_dispatch_
; CHECK: ret void
