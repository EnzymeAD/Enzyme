; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Vector forward mode (width 3) through an OpenMP parallel loop. A shadow is
; an array of the lanes' shadows, which cannot pass through the variadic fork
; call: a trampoline takes the lanes separately and calls the derivative.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

define void @caller(ptr %x, ptr %dx, i64 %n) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @square, metadata !"enzyme_width", i64 3, metadata !"enzyme_dupv", i64 8, ptr %x, ptr %dx, i64 %n)
  ret void
}

declare void @__enzyme_fwddiff(...)

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
  %cmp.not = icmp eq i64 %n, 0
  br i1 %cmp.not, label %omp.precond.end, label %omp.precond.then

omp.precond.then:
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
  %iv = phi i64 [ %iv.next, %omp.inner.for.body ], [ %lb, %omp.precond.then ]
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %iv
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %arrayidx, align 8
  %iv.next = add nuw i64 %iv, 1
  %cmp7 = icmp ult i64 %iv.next, %add29
  br i1 %cmp7, label %omp.inner.for.body, label %omp.loop.exit

omp.loop.exit:
  call void @__kmpc_for_static_fini(ptr nonnull @1, i32 %tid)
  br label %omp.precond.end

omp.precond.end:
  ret void
}

declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)
declare void @__kmpc_for_static_fini(ptr, i32)
declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}



; CHECK: define internal void @fwddiffe3square(ptr %x, [3 x ptr] %"x'", i64 %n)
; CHECK:   %[[a:.+]] = extractvalue [3 x ptr] %"x'", 0
; CHECK:   %[[b:.+]] = extractvalue [3 x ptr] %"x'", 1
; CHECK:   %[[c:.+]] = extractvalue [3 x ptr] %"x'", 2
; CHECK:   call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @2, i32 5, ptr @fwddiffe3.omp_outlined..lanes, i64 %n, ptr %x, ptr %[[a]], ptr %[[b]], ptr %[[c]])

; CHECK: define internal void @fwddiffe3.omp_outlined..lanes(ptr {{.*}}%0, ptr {{.*}}%1, i64 %2, ptr %3, ptr %4, ptr %5, ptr %6)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %7 = insertvalue [3 x ptr] undef, ptr %4, 0
; CHECK-NEXT:   %8 = insertvalue [3 x ptr] %7, ptr %5, 1
; CHECK-NEXT:   %9 = insertvalue [3 x ptr] %8, ptr %6, 2
; CHECK-NEXT:   call void @fwddiffe3.omp_outlined.(ptr %0, ptr %1, i64 %2, ptr %3, [3 x ptr] %9)
; CHECK-NEXT:   ret void
