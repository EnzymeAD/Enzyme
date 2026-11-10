; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: combined_ok max_after=4 rev_forks=5 d0=6.000000 d4=30.000000
; EVAL-NEXT: split_ok max_mid=3 max_after=6 d0=12.000000 d2=20.000000

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str.loc = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@loc = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @.str.loc }, align 8

@.fmt_comb = private unnamed_addr constant [51 x i8] c"combined_ok max_after=%d rev_forks=%d d0=%f d4=%f\0A\00", align 1
@.fmt_split = private unnamed_addr constant [46 x i8] c"split_ok max_mid=%d max_after=%d d0=%f d2=%f\0A\00", align 1

@g_max_threads = internal global i32 4, align 4
@g_pushed_threads = internal global i32 0, align 4
@g_tid = internal global i32 0, align 4
@g_last_fork_threads = internal global i32 0, align 4

declare void @__enzyme_autodiff(...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)
declare i32 @printf(ptr, ...)
declare void @abort()
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define i64 @omp_get_max_threads() noinline {
entry:
  %v = load i32, ptr @g_max_threads, align 4
  %ext = sext i32 %v to i64
  ret i64 %ext
}

define void @omp_set_num_threads(i32 %n) noinline {
entry:
  %pos = icmp sgt i32 %n, 0
  br i1 %pos, label %set, label %exit

set:
  store i32 %n, ptr @g_max_threads, align 4
  br label %exit

exit:
  ret void
}

define i64 @omp_get_thread_num() noinline {
entry:
  %tid = load i32, ptr @g_tid, align 4
  %ext = sext i32 %tid to i64
  ret i64 %ext
}

define i32 @__kmpc_global_thread_num(ptr %ident) noinline {
entry:
  %tid = load i32, ptr @g_tid, align 4
  ret i32 %tid
}

define void @__kmpc_push_num_threads(ptr %ident, i32 %gtid, i32 %num_threads) noinline {
entry:
  store i32 %num_threads, ptr @g_pushed_threads, align 4
  ret void
}

define void @__kmpc_fork_call(ptr %ident, i32 %argc, ptr %microtask, ...) noinline {
entry:
  %pushed = load i32, ptr @g_pushed_threads, align 4
  store i32 0, ptr @g_pushed_threads, align 4
  %max_t = load i32, ptr @g_max_threads, align 4
  %has_push = icmp sgt i32 %pushed, 0
  %nthreads = select i1 %has_push, i32 %pushed, i32 %max_t
  store i32 %nthreads, ptr @g_last_fork_threads, align 4

  %args_arr = alloca [8 x ptr], align 8
  %ap = alloca [64 x i8], align 16
  call void @llvm.va_start(ptr %ap)
  br label %va_loop

va_loop:
  %idx = phi i32 [ 0, %entry ], [ %idx_next, %va_body ]
  %more = icmp slt i32 %idx, %argc
  br i1 %more, label %va_body, label %va_done

va_body:
  %val = va_arg ptr %ap, ptr
  %slot = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 %idx
  store ptr %val, ptr %slot, align 8
  %idx_next = add i32 %idx, 1
  br label %va_loop

va_done:
  call void @llvm.va_end(ptr %ap)
  %gtid_slot = alloca i32, align 4
  %btid_slot = alloca i32, align 4
  br label %t_loop

t_loop:
  %tid = phi i32 [ 0, %va_done ], [ %tid_next, %t_cont ]
  %t_more = icmp slt i32 %tid, %nthreads
  br i1 %t_more, label %t_body, label %t_done

t_body:
  store i32 %tid, ptr @g_tid, align 4
  store i32 %tid, ptr %gtid_slot, align 4
  store i32 %tid, ptr %btid_slot, align 4
  %p0 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 0
  %a0 = load ptr, ptr %p0, align 8
  %p1 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 1
  %a1 = load ptr, ptr %p1, align 8
  %p2 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 2
  %a2 = load ptr, ptr %p2, align 8
  %p3 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 3
  %a3 = load ptr, ptr %p3, align 8
  %p4 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 4
  %a4 = load ptr, ptr %p4, align 8
  switch i32 %argc, label %bad_argc [
    i32 2, label %call2
    i32 4, label %call4
    i32 5, label %call5
  ]

bad_argc:
  call void @abort()
  unreachable

call2:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1)
  br label %t_cont

call4:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3)
  br label %t_cont

call5:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4)
  br label %t_cont

t_cont:
  %tid_next = add i32 %tid, 1
  br label %t_loop

t_done:
  store i32 0, ptr @g_tid, align 4
  ret void
}

define internal void @outlined_overwritten(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture align 8 %arr, ptr nocapture align 8 %out) {
entry:
  %gtid = load i32, ptr %gtid_ptr, align 4
  %idx = sext i32 %gtid to i64
  %aptr = getelementptr inbounds double, ptr %arr, i64 %idx
  %optr = getelementptr inbounds double, ptr %out, i64 %idx
  %v = load double, ptr %aptr, align 8
  ; Overwrite arr[idx] so Enzyme caches v per thread on the fork tape.
  store double 0.000000e+00, ptr %aptr, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %optr, align 8
  ret void
}

define void @set_and_push_kernel(ptr %arr, ptr %out) {
entry:
  call void @omp_set_num_threads(i32 2)
  %gtid = call i32 @__kmpc_global_thread_num(ptr @loc)
  ; Push 5 threads even though omp_set_num_threads lowered max_threads to 2.
  call void @__kmpc_push_num_threads(ptr @loc, i32 %gtid, i32 5)
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 2, ptr @outlined_overwritten, ptr %arr, ptr %out)
  ret void
}

define void @set_only_kernel(ptr %arr, ptr %out) {
entry:
  call void @omp_set_num_threads(i32 3)
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 2, ptr @outlined_overwritten, ptr %arr, ptr %out)
  ret void
}

define i32 @main() {
entry:
  %arr = alloca [5 x double], align 8
  %d_arr = alloca [5 x double], align 8
  %out = alloca [5 x double], align 8
  %d_out = alloca [5 x double], align 8
  br label %init_comb

init_comb:
  %i = phi i64 [ 0, %entry ], [ %i_next, %init_comb ]
  %pa = getelementptr inbounds [5 x double], ptr %arr, i64 0, i64 %i
  %pda = getelementptr inbounds [5 x double], ptr %d_arr, i64 0, i64 %i
  %po = getelementptr inbounds [5 x double], ptr %out, i64 0, i64 %i
  %pdo = getelementptr inbounds [5 x double], ptr %d_out, i64 0, i64 %i
  %ip1 = add i64 %i, 1
  %val = sitofp i64 %ip1 to double
  store double %val, ptr %pa, align 8
  store double 0.000000e+00, ptr %pda, align 8
  store double 0.000000e+00, ptr %po, align 8
  store double 3.000000e+00, ptr %pdo, align 8
  %i_next = add i64 %i, 1
  %cond = icmp slt i64 %i_next, 5
  br i1 %cond, label %init_comb, label %run_comb

run_comb:
  store i32 4, ptr @g_max_threads, align 4
  store i32 0, ptr @g_last_fork_threads, align 4
  call void (...) @__enzyme_autodiff(ptr @set_and_push_kernel, metadata !"enzyme_dup", ptr %arr, ptr %d_arr, metadata !"enzyme_dup", ptr %out, ptr %d_out)
  %max_after_comb = load i32, ptr @g_max_threads, align 4
  %rev_forks = load i32, ptr @g_last_fork_threads, align 4
  %da0_ptr = getelementptr inbounds [5 x double], ptr %d_arr, i64 0, i64 0
  %da4_ptr = getelementptr inbounds [5 x double], ptr %d_arr, i64 0, i64 4
  %da0 = load double, ptr %da0_ptr, align 8
  %da4 = load double, ptr %da4_ptr, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt_comb, i32 %max_after_comb, i32 %rev_forks, double %da0, double %da4)
  br label %init_split

init_split:
  %j = phi i64 [ 0, %run_comb ], [ %j_next, %init_split ]
  %spa = getelementptr inbounds [5 x double], ptr %arr, i64 0, i64 %j
  %spda = getelementptr inbounds [5 x double], ptr %d_arr, i64 0, i64 %j
  %spo = getelementptr inbounds [5 x double], ptr %out, i64 0, i64 %j
  %spdo = getelementptr inbounds [5 x double], ptr %d_out, i64 0, i64 %j
  %jp2 = add i64 %j, 3
  %sval = sitofp i64 %jp2 to double
  store double %sval, ptr %spa, align 8
  store double 0.000000e+00, ptr %spda, align 8
  store double 0.000000e+00, ptr %spo, align 8
  store double 2.000000e+00, ptr %spdo, align 8
  %j_next = add i64 %j, 1
  %scond = icmp slt i64 %j_next, 5
  br i1 %scond, label %init_split, label %run_split

run_split:
  store i32 6, ptr @g_max_threads, align 4
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @set_only_kernel, metadata !"enzyme_dup", ptr %arr, ptr %d_arr, metadata !"enzyme_dup", ptr %out, ptr %d_out)
  %max_mid = load i32, ptr @g_max_threads, align 4
  call void (...) @__enzyme_reverse(ptr @set_only_kernel, metadata !"enzyme_dup", ptr %arr, ptr %d_arr, metadata !"enzyme_dup", ptr %out, ptr %d_out, ptr %tape)
  %max_after_split = load i32, ptr @g_max_threads, align 4
  %sda0 = load double, ptr %da0_ptr, align 8
  %da2_ptr = getelementptr inbounds [5 x double], ptr %d_arr, i64 0, i64 2
  %sda2 = load double, ptr %da2_ptr, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt_split, i32 %max_mid, i32 %max_after_split, double %sda0, double %sda2)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffeset_and_push_kernel(
; CHECK: %[[PREV_MAX64:.+]] = call i64 @omp_get_max_threads()
; CHECK: %[[PREV_MAX:.+]] = trunc i64 %[[PREV_MAX64]] to i32
; CHECK: call void @omp_set_num_threads(i32 2)
; CHECK: call void @__kmpc_push_num_threads(ptr @loc, i32 %{{.+}}, i32 5)
; CHECK: %[[UMAX:.+]] = call i64 @llvm.umax.i64(i64 %{{.+}}, i64 5)
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 5, ptr @augmented_outlined_overwritten
; CHECK: call void @__kmpc_push_num_threads(ptr @loc, i32 %{{.+}}, i32 5)
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 5, ptr @diffeoutlined_overwritten
; CHECK: call void @omp_set_num_threads(i32 %[[PREV_MAX]])
