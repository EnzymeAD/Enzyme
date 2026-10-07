; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; flang's OpenMPIRBuilder passes the variables a parallel region captures in
; one struct. Enzyme passes them to a copy of the outlined function as
; separate arguments, so that the inactive bound %n is not passed with a
; shadow, and marks the ones only read as readonly and nocapture.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

define void @caller(ptr %x, ptr %dx, ptr %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @square, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %n)
  ret void
}

declare void @__enzyme_autodiff(...)

define internal void @square(ptr %x, ptr %n) {
entry:
  %structArg = alloca { ptr, ptr }, align 8
  store ptr %n, ptr %structArg, align 8
  %gep = getelementptr inbounds i8, ptr %structArg, i64 8
  store ptr %x, ptr %gep, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @2, i32 1, ptr @square..omp_par, ptr nonnull %structArg)
  ret void
}

define internal void @square..omp_par(ptr noalias nocapture readnone %tid.addr, ptr noalias nocapture readnone %zero.addr, ptr nocapture readonly %0) {
omp.par.entry:
  %loadn = load ptr, ptr %0, align 8
  %gep1 = getelementptr i8, ptr %0, i64 8
  %loadx = load ptr, ptr %gep1, align 8
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  %nv = load i64, ptr %loadn, align 8
  %ub0 = add i64 %nv, -1
  store i64 0, ptr %p.lowerbound, align 8
  store i64 %ub0, ptr %p.upperbound, align 8
  store i64 1, ptr %p.stride, align 8
  %tid = call i32 @__kmpc_global_thread_num(ptr nonnull @1)
  call void @__kmpc_for_static_init_8u(ptr nonnull @1, i32 %tid, i32 34, ptr nonnull %p.lastiter, ptr nonnull %p.lowerbound, ptr nonnull %p.upperbound, ptr nonnull %p.stride, i64 1, i64 0)
  %lb = load i64, ptr %p.lowerbound, align 8
  %ub = load i64, ptr %p.upperbound, align 8
  %trip = sub i64 %ub, %lb
  %count = add i64 %trip, 1
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %exit, label %body

body:
  %iv = phi i64 [ 0, %omp.par.entry ], [ %iv.next, %body ]
  %i = add i64 %iv, %lb
  %arrayidx = getelementptr inbounds double, ptr %loadx, i64 %i
  %v = load double, ptr %arrayidx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %arrayidx, align 8
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %count
  br i1 %done, label %exit, label %body

exit:
  call void @__kmpc_for_static_fini(ptr nonnull @1, i32 %tid)
  call void @__kmpc_barrier(ptr nonnull @1, i32 %tid)
  ret void
}

declare i32 @__kmpc_global_thread_num(ptr)
declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)
declare void @__kmpc_for_static_fini(ptr, i32)
declare void @__kmpc_barrier(ptr, i32)
declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @square..omp_par.enzyme_unpacked.0.8(ptr {{.*}}%tid.addr, ptr {{.*}}%zero.addr, ptr {{(nocapture readonly|readonly captures\(none\))}} %.0, ptr {{(nocapture|captures\(none\))}} %.8)
; CHECK-NOT: %structArg

; CHECK: define internal void @diffesquare(ptr %x, ptr %"x'", ptr %n)
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @2, i32 4, ptr @augmented_square..omp_par.enzyme_unpacked.0.8{{.*}}, ptr %n, ptr %x, ptr %"x'", ptr
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @2, i32 4, ptr @diffesquare..omp_par.enzyme_unpacked.0.8, ptr %n, ptr %x, ptr %"x'", ptr
