; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: mode2_ok d0=-4.000000 d7=38.000000 dinit=3.000000,5.000000 crit=0
; EVAL-NEXT: mode1_ok d0=-4.000000 d7=38.000000 crit=4 depth=0
; EVAL-NEXT: block2_ok count=1 d0=2.000000 d5=16.000000 barriers=8 crit=0 depth=0
; EVAL-NEXT: block1_ok count=1 d0=2.000000 d5=16.000000 barriers=8 crit=4 depth=0

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str.loc = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@loc = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @.str.loc }, align 8
@loc_for = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @.str.loc }, align 8
@loc_red = private unnamed_addr constant %struct.ident_t { i32 0, i32 18, i32 0, i32 22, ptr @.str.loc }, align 8
@orig_lock = internal global [8 x i32] zeroinitializer, align 8

@.fmt_m2 = private unnamed_addr constant [42 x i8] c"mode2_ok d0=%f d7=%f dinit=%f,%f crit=%d\0A\00", align 1
@.fmt_m1 = private unnamed_addr constant [39 x i8] c"mode1_ok d0=%f d7=%f crit=%d depth=%d\0A\00", align 1
@.fmt_blk = private unnamed_addr constant [62 x i8] c"block%d_ok count=%d d0=%f d5=%f barriers=%d crit=%d depth=%d\0A\00", align 1

@enzyme_dup = external global i32, align 4
@enzyme_const = external global i32, align 4

@g_tid = internal global i32 0, align 4
@g_nthreads = internal global i32 1, align 4
@g_reduce_mode = internal global i32 2, align 4
@g_crit_depth = internal global i32 0, align 4
@g_crit_acquires = internal global i32 0, align 4
@g_barriers = internal global i32 0, align 4

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
declare void @abort()
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define i64 @omp_get_max_threads() noinline {
entry:
  ret i64 4
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

define void @__kmpc_for_static_init_4(ptr %ident, i32 %gtid, i32 %sched, ptr %plast, ptr %plb, ptr %pub, ptr %pstride, i32 %incr, i32 %chunk) noinline {
entry:
  %lb = load i32, ptr %plb, align 4
  %ub = load i32, ptr %pub, align 4
  %span = sub i32 %ub, %lb
  %count = add i32 %span, 1
  %empty = icmp sle i32 %count, 0
  br i1 %empty, label %none, label %split

none:
  store i32 1, ptr %plb, align 4
  store i32 0, ptr %pub, align 4
  store i32 0, ptr %plast, align 4
  ret void

split:
  %n = load i32, ptr @g_nthreads, align 4
  %base = sdiv i32 %count, %n
  %rem = srem i32 %count, %n
  %lt = icmp slt i32 %gtid, %rem
  %off_extra = select i1 %lt, i32 %gtid, i32 %rem
  %len_extra = select i1 %lt, i32 1, i32 0
  %mul = mul i32 %gtid, %base
  %start_off = add i32 %mul, %off_extra
  %start = add i32 %lb, %start_off
  %len = add i32 %base, %len_extra
  %end_excl = add i32 %start, %len
  %my_ub = sub i32 %end_excl, 1
  %has_work = icmp sgt i32 %len, 0
  %out_lb = select i1 %has_work, i32 %start, i32 1
  %out_ub = select i1 %has_work, i32 %my_ub, i32 0
  store i32 %out_lb, ptr %plb, align 4
  store i32 %out_ub, ptr %pub, align 4
  store i32 %count, ptr %pstride, align 4
  %is_last_ub = icmp eq i32 %my_ub, %ub
  %is_last = and i1 %has_work, %is_last_ub
  %last_i32 = zext i1 %is_last to i32
  store i32 %last_i32, ptr %plast, align 4
  ret void
}

define void @__kmpc_for_static_fini(ptr %ident, i32 %gtid) noinline {
entry:
  ret void
}

define void @__kmpc_barrier(ptr %ident, i32 %gtid) noinline {
entry:
  %old = load i32, ptr @g_barriers, align 4
  %inc = add i32 %old, 1
  store i32 %inc, ptr @g_barriers, align 4
  ret void
}

define void @__kmpc_critical(ptr %ident, i32 %gtid, ptr %crit) noinline {
entry:
  %d = load i32, ptr @g_crit_depth, align 4
  %bad = icmp ne i32 %d, 0
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  store i32 1, ptr @g_crit_depth, align 4
  %c = load i32, ptr @g_crit_acquires, align 4
  %c1 = add i32 %c, 1
  store i32 %c1, ptr @g_crit_acquires, align 4
  ret void
}

define void @__kmpc_end_critical(ptr %ident, i32 %gtid, ptr %crit) noinline {
entry:
  %d = load i32, ptr @g_crit_depth, align 4
  %ok_cond = icmp eq i32 %d, 1
  br i1 %ok_cond, label %ok, label %fail

fail:
  call void @abort()
  unreachable

ok:
  store i32 0, ptr @g_crit_depth, align 4
  ret void
}

define i32 @__kmpc_reduce_nowait(ptr %ident, i32 %gtid, i32 %nvars, i64 %size, ptr %data, ptr %func, ptr %lck) noinline {
entry:
  %d_null = icmp eq ptr %data, null
  %f_null = icmp eq ptr %func, null
  %both_null = and i1 %d_null, %f_null
  br i1 %both_null, label %ok, label %fail

fail:
  call void @abort()
  unreachable

ok:
  %m = load i32, ptr @g_reduce_mode, align 4
  ret i32 %m
}

define void @__kmpc_end_reduce_nowait(ptr %ident, i32 %gtid, ptr %lck) noinline {
entry:
  ret void
}

define i32 @__kmpc_reduce(ptr %ident, i32 %gtid, i32 %nvars, i64 %size, ptr %data, ptr %func, ptr %lck) noinline {
entry:
  %r = call i32 @__kmpc_reduce_nowait(ptr %ident, i32 %gtid, i32 %nvars, i64 %size, ptr %data, ptr %func, ptr %lck)
  ret i32 %r
}

define void @__kmpc_end_reduce(ptr %ident, i32 %gtid, ptr %lck) noinline {
entry:
  ret void
}

define internal void @dummy_tree_reducer(ptr %lhs, ptr %rhs) noinline {
entry:
  call void @abort()
  unreachable
}

define void @__kmpc_fork_call(ptr %ident, i32 %argc, ptr %microtask, ...) noinline {
entry:
  store i32 4, ptr @g_nthreads, align 4
  %args_arr = alloca [12 x ptr], align 8
  %ap = alloca [64 x i8], align 16
  call void @llvm.va_start(ptr %ap)
  br label %va_loop

va_loop:
  %idx = phi i32 [ 0, %entry ], [ %idx_next, %va_body ]
  %more = icmp slt i32 %idx, %argc
  br i1 %more, label %va_body, label %va_done

va_body:
  %val = va_arg ptr %ap, ptr
  %slot = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 %idx
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
  %t_more = icmp slt i32 %tid, 4
  br i1 %t_more, label %t_body, label %t_done

t_body:
  store i32 %tid, ptr @g_tid, align 4
  store i32 %tid, ptr %gtid_slot, align 4
  store i32 %tid, ptr %btid_slot, align 4
  %p0 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 0
  %a0 = load ptr, ptr %p0, align 8
  %p1 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 1
  %a1 = load ptr, ptr %p1, align 8
  %p2 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 2
  %a2 = load ptr, ptr %p2, align 8
  %p3 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 3
  %a3 = load ptr, ptr %p3, align 8
  %p4 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 4
  %a4 = load ptr, ptr %p4, align 8
  %p5 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 5
  %a5 = load ptr, ptr %p5, align 8
  %p6 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 6
  %a6 = load ptr, ptr %p6, align 8
  %p7 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 7
  %a7 = load ptr, ptr %p7, align 8
  %p8 = getelementptr inbounds [12 x ptr], ptr %args_arr, i32 0, i32 8
  %a8 = load ptr, ptr %p8, align 8
  switch i32 %argc, label %bad_argc [
    i32 4, label %call4
    i32 6, label %call6
    i32 7, label %call7
    i32 8, label %call8
    i32 9, label %call9
  ]

bad_argc:
  call void @abort()
  unreachable

call4:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3)
  br label %t_cont

call6:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5)
  br label %t_cont

call7:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6)
  br label %t_cont

call8:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7)
  br label %t_cont

call9:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8)
  br label %t_cont

t_cont:
  %tid_next = add i32 %tid, 1
  br label %t_loop

t_done:
  store i32 0, ptr @g_tid, align 4
  store i32 1, ptr @g_nthreads, align 4
  ret void
}

define internal void @outlined_plus_minus(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture align 8 %sum_ptr, ptr nocapture align 8 %diff_ptr, ptr nocapture readonly align 4 %count_ptr, ptr nocapture readonly align 8 %arr) {
entry:
  %lb = alloca i32, align 4
  %ub = alloca i32, align 4
  %stride = alloca i32, align 4
  %last = alloca i32, align 4
  %local_sum = alloca double, align 8
  %local_diff = alloca double, align 8
  %red_list = alloca [2 x ptr], align 8

  %count = load i32, ptr %count_ptr, align 4
  %gtid = load i32, ptr %gtid_ptr, align 4
  %upper_init = sub i32 %count, 1
  store i32 0, ptr %lb, align 4
  store i32 %upper_init, ptr %ub, align 4
  store i32 1, ptr %stride, align 4
  store i32 0, ptr %last, align 4
  store double 0.000000e+00, ptr %local_sum, align 8
  store double 0.000000e+00, ptr %local_diff, align 8

  call void @__kmpc_for_static_init_4(ptr @loc_for, i32 %gtid, i32 34, ptr %last, ptr %lb, ptr %ub, ptr %stride, i32 1, i32 1)
  %my_lb = load i32, ptr %lb, align 4
  %my_ub = load i32, ptr %ub, align 4
  %has_iters = icmp sle i32 %my_lb, %my_ub
  br i1 %has_iters, label %loop, label %after_loop

loop:
  %iv = phi i32 [ %my_lb, %entry ], [ %iv_next, %loop ]
  %idx64 = sext i32 %iv to i64
  %elem_ptr = getelementptr inbounds double, ptr %arr, i64 %idx64
  %val = load double, ptr %elem_ptr, align 8
  %sq = fmul double %val, %val
  %cur_s = load double, ptr %local_sum, align 8
  %next_s = fadd double %cur_s, %sq
  store double %next_s, ptr %local_sum, align 8
  %two_val = fmul double 2.000000e+00, %val
  %cur_d = load double, ptr %local_diff, align 8
  %next_d = fsub double %cur_d, %two_val
  store double %next_d, ptr %local_diff, align 8
  %iv_next = add nsw i32 %iv, 1
  %cont = icmp sle i32 %iv_next, %my_ub
  br i1 %cont, label %loop, label %after_loop

after_loop:
  call void @__kmpc_for_static_fini(ptr @loc_for, i32 %gtid)
  %rl0 = getelementptr inbounds [2 x ptr], ptr %red_list, i64 0, i64 0
  store ptr %local_sum, ptr %rl0, align 8
  %rl1 = getelementptr inbounds [2 x ptr], ptr %red_list, i64 0, i64 1
  store ptr %local_diff, ptr %rl1, align 8
  %red_kind = call i32 @__kmpc_reduce_nowait(ptr @loc_red, i32 %gtid, i32 2, i64 16, ptr %red_list, ptr @dummy_tree_reducer, ptr @orig_lock)
  switch i32 %red_kind, label %red_done [
    i32 1, label %red_crit
    i32 2, label %red_atomic
  ]

red_crit:
  %s_glob = load double, ptr %sum_ptr, align 8
  %s_loc = load double, ptr %local_sum, align 8
  %s_comb = fadd double %s_glob, %s_loc
  store double %s_comb, ptr %sum_ptr, align 8
  %d_glob = load double, ptr %diff_ptr, align 8
  %d_loc = load double, ptr %local_diff, align 8
  %d_comb = fadd double %d_glob, %d_loc
  store double %d_comb, ptr %diff_ptr, align 8
  call void @__kmpc_end_reduce_nowait(ptr @loc_red, i32 %gtid, ptr @orig_lock)
  br label %red_done

red_atomic:
  %s_loc_a = load double, ptr %local_sum, align 8
  %old_s = atomicrmw fadd ptr %sum_ptr, double %s_loc_a monotonic, align 8
  %d_loc_a = load double, ptr %local_diff, align 8
  %old_d = atomicrmw fadd ptr %diff_ptr, double %d_loc_a monotonic, align 8
  br label %red_done

red_done:
  ret void
}

define double @plus_minus_reduce(ptr %arr, i32 %count, double %init_sum, double %init_diff) {
entry:
  %sum_slot = alloca double, align 8
  %diff_slot = alloca double, align 8
  %count_slot = alloca i32, align 4
  store double %init_sum, ptr %sum_slot, align 8
  store double %init_diff, ptr %diff_slot, align 8
  store i32 %count, ptr %count_slot, align 4
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 4, ptr @outlined_plus_minus, ptr %sum_slot, ptr %diff_slot, ptr %count_slot, ptr %arr)
  %final_sum = load double, ptr %sum_slot, align 8
  %final_diff = load double, ptr %diff_slot, align 8
  %term_s = fmul double 3.000000e+00, %final_sum
  %term_d = fmul double 5.000000e+00, %final_diff
  %res = fadd double %term_s, %term_d
  ret double %res
}

define internal void @outlined_blocking_mixed(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture align 8 %sum_ptr, ptr nocapture align 4 %cnt_ptr, ptr nocapture readonly align 4 %count_ptr, ptr nocapture readonly align 8 %arr) {
entry:
  %lb = alloca i32, align 4
  %ub = alloca i32, align 4
  %stride = alloca i32, align 4
  %last = alloca i32, align 4
  %local_sum = alloca double, align 8
  %local_cnt = alloca i32, align 4
  %red_list = alloca [2 x ptr], align 8

  %count = load i32, ptr %count_ptr, align 4
  %gtid = load i32, ptr %gtid_ptr, align 4
  %upper_init = sub i32 %count, 1
  store i32 0, ptr %lb, align 4
  store i32 %upper_init, ptr %ub, align 4
  store i32 1, ptr %stride, align 4
  store i32 0, ptr %last, align 4
  store double 0.000000e+00, ptr %local_sum, align 8
  store i32 0, ptr %local_cnt, align 4

  call void @__kmpc_for_static_init_4(ptr @loc_for, i32 %gtid, i32 34, ptr %last, ptr %lb, ptr %ub, ptr %stride, i32 1, i32 1)
  %my_lb = load i32, ptr %lb, align 4
  %my_ub = load i32, ptr %ub, align 4
  %has_iters = icmp sle i32 %my_lb, %my_ub
  br i1 %has_iters, label %loop, label %after_loop

loop:
  %iv = phi i32 [ %my_lb, %entry ], [ %iv_next, %loop ]
  %idx64 = sext i32 %iv to i64
  %elem_ptr = getelementptr inbounds double, ptr %arr, i64 %idx64
  %val = load double, ptr %elem_ptr, align 8
  %gt5 = fcmp ogt double %val, 5.000000e+00
  %inc_c = zext i1 %gt5 to i32
  %cur_c = load i32, ptr %local_cnt, align 4
  %new_c = add i32 %cur_c, %inc_c
  store i32 %new_c, ptr %local_cnt, align 4

  %sq = fmul double %val, %val
  %cur_s = load double, ptr %local_sum, align 8
  %next_s = fadd double %cur_s, %sq
  store double %next_s, ptr %local_sum, align 8
  %iv_next = add nsw i32 %iv, 1
  %cont = icmp sle i32 %iv_next, %my_ub
  br i1 %cont, label %loop, label %after_loop

after_loop:
  call void @__kmpc_for_static_fini(ptr @loc_for, i32 %gtid)
  %rl0 = getelementptr inbounds [2 x ptr], ptr %red_list, i64 0, i64 0
  store ptr %local_sum, ptr %rl0, align 8
  %rl1 = getelementptr inbounds [2 x ptr], ptr %red_list, i64 0, i64 1
  store ptr %local_cnt, ptr %rl1, align 8
  %red_kind = call i32 @__kmpc_reduce(ptr @loc_red, i32 %gtid, i32 2, i64 16, ptr %red_list, ptr @dummy_tree_reducer, ptr @orig_lock)
  switch i32 %red_kind, label %red_done [
    i32 1, label %red_crit
    i32 2, label %red_atomic
  ]

red_crit:
  %s_glob = load double, ptr %sum_ptr, align 8
  %s_loc = load double, ptr %local_sum, align 8
  %s_comb = fadd double %s_glob, %s_loc
  store double %s_comb, ptr %sum_ptr, align 8
  %c_glob = load i32, ptr %cnt_ptr, align 4
  %c_loc = load i32, ptr %local_cnt, align 4
  %c_comb = add i32 %c_glob, %c_loc
  store i32 %c_comb, ptr %cnt_ptr, align 4
  br label %red_end

red_atomic:
  %s_loc_a = load double, ptr %local_sum, align 8
  %old_s = atomicrmw fadd ptr %sum_ptr, double %s_loc_a monotonic, align 8
  %c_loc_a = load i32, ptr %local_cnt, align 4
  %old_c = atomicrmw add ptr %cnt_ptr, i32 %c_loc_a monotonic, align 4
  br label %red_end

red_end:
  call void @__kmpc_end_reduce(ptr @loc_red, i32 %gtid, ptr @orig_lock)
  br label %red_done

red_done:
  ret void
}

define double @blocking_mixed_reduce(ptr %arr, i32 %count, i32 %init_cnt, ptr %cnt_out) {
entry:
  %sum_slot = alloca double, align 8
  %cnt_slot = alloca i32, align 4
  %count_slot = alloca i32, align 4
  store double 0.000000e+00, ptr %sum_slot, align 8
  store i32 %init_cnt, ptr %cnt_slot, align 4
  store i32 %count, ptr %count_slot, align 4
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 4, ptr @outlined_blocking_mixed, ptr %sum_slot, ptr %cnt_slot, ptr %count_slot, ptr %arr)
  %c_val = load i32, ptr %cnt_slot, align 4
  store i32 %c_val, ptr %cnt_out, align 4
  %s_val = load double, ptr %sum_slot, align 8
  ret double %s_val
}

define i32 @main() {
entry:
  %arr8 = alloca [8 x double], align 8
  %d_arr8 = alloca [8 x double], align 8
  %scratch8 = alloca [8 x double], align 8
  br label %init8

init8:
  %i8 = phi i64 [ 0, %entry ], [ %i8_next, %init8 ]
  %p_a8 = getelementptr inbounds [8 x double], ptr %arr8, i64 0, i64 %i8
  %p_d8 = getelementptr inbounds [8 x double], ptr %d_arr8, i64 0, i64 %i8
  %p_s8 = getelementptr inbounds [8 x double], ptr %scratch8, i64 0, i64 %i8
  %i8_plus1 = add i64 %i8, 1
  %v8 = sitofp i64 %i8_plus1 to double
  store double %v8, ptr %p_a8, align 8
  store double 0.000000e+00, ptr %p_d8, align 8
  store double 0.000000e+00, ptr %p_s8, align 8
  %i8_next = add i64 %i8, 1
  %c8 = icmp slt i64 %i8_next, 8
  br i1 %c8, label %init8, label %run_m2

run_m2:
  %dup_code = load i32, ptr @enzyme_dup, align 4
  %const_code = load i32, ptr @enzyme_const, align 4
  store i32 2, ptr @g_reduce_mode, align 4
  store i32 0, ptr @g_crit_acquires, align 4
  %d_inits = call double (...) @__enzyme_autodiff(ptr @plus_minus_reduce, i32 %dup_code, ptr %arr8, ptr %d_arr8, i32 %const_code, i32 8, double 7.000000e+00, i32 %const_code, double -4.000000e+00)
  %d_initd = call double (...) @__enzyme_autodiff(ptr @plus_minus_reduce, i32 %dup_code, ptr %arr8, ptr %scratch8, i32 %const_code, i32 8, i32 %const_code, double 7.000000e+00, double -4.000000e+00)
  %d0_m2_ptr = getelementptr inbounds [8 x double], ptr %d_arr8, i64 0, i64 0
  %d7_m2_ptr = getelementptr inbounds [8 x double], ptr %d_arr8, i64 0, i64 7
  %d0_m2 = load double, ptr %d0_m2_ptr, align 8
  %d7_m2 = load double, ptr %d7_m2_ptr, align 8
  %crit_m2 = load i32, ptr @g_crit_acquires, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_m2, double %d0_m2, double %d7_m2, double %d_inits, double %d_initd, i32 %crit_m2)
  br label %zero8

zero8:
  %z8 = phi i64 [ 0, %run_m2 ], [ %z8_next, %zero8 ]
  %pz8 = getelementptr inbounds [8 x double], ptr %d_arr8, i64 0, i64 %z8
  store double 0.000000e+00, ptr %pz8, align 8
  %z8_next = add i64 %z8, 1
  %cz8 = icmp slt i64 %z8_next, 8
  br i1 %cz8, label %zero8, label %run_m1

run_m1:
  store i32 1, ptr @g_reduce_mode, align 4
  store i32 0, ptr @g_crit_acquires, align 4
  %ign1 = call double (...) @__enzyme_autodiff(ptr @plus_minus_reduce, i32 %dup_code, ptr %arr8, ptr %d_arr8, i32 %const_code, i32 8, i32 %const_code, double 7.000000e+00, i32 %const_code, double -4.000000e+00)
  %d0_m1 = load double, ptr %d0_m2_ptr, align 8
  %d7_m1 = load double, ptr %d7_m2_ptr, align 8
  %crit_m1 = load i32, ptr @g_crit_acquires, align 4
  %depth_m1 = load i32, ptr @g_crit_depth, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_m1, double %d0_m1, double %d7_m1, i32 %crit_m1, i32 %depth_m1)

  %arr6 = alloca [6 x double], align 8
  %d_arr6 = alloca [6 x double], align 8
  %cnt_box = alloca i32, align 4
  br label %init6

init6:
  %i6 = phi i64 [ 0, %run_m1 ], [ %i6_next, %init6 ]
  %p_a6 = getelementptr inbounds [6 x double], ptr %arr6, i64 0, i64 %i6
  %p_d6 = getelementptr inbounds [6 x double], ptr %d_arr6, i64 0, i64 %i6
  %is5 = icmp eq i64 %i6, 5
  %i6_plus1 = add i64 %i6, 1
  %base6 = sitofp i64 %i6_plus1 to double
  %v6 = select i1 %is5, double 8.000000e+00, double %base6
  store double %v6, ptr %p_a6, align 8
  store double 0.000000e+00, ptr %p_d6, align 8
  %i6_next = add i64 %i6, 1
  %c6 = icmp slt i64 %i6_next, 6
  br i1 %c6, label %init6, label %run_b2

run_b2:
  store i32 2, ptr @g_reduce_mode, align 4
  store i32 0, ptr @g_barriers, align 4
  store i32 0, ptr @g_crit_acquires, align 4
  store i32 0, ptr %cnt_box, align 4
  %ign2 = call double (...) @__enzyme_autodiff(ptr @blocking_mixed_reduce, i32 %dup_code, ptr %arr6, ptr %d_arr6, i32 %const_code, i32 6, i32 %const_code, i32 0, i32 %const_code, ptr %cnt_box)
  %cnt_b2 = load i32, ptr %cnt_box, align 4
  %d0_b_ptr = getelementptr inbounds [6 x double], ptr %d_arr6, i64 0, i64 0
  %d5_b_ptr = getelementptr inbounds [6 x double], ptr %d_arr6, i64 0, i64 5
  %d0_b2 = load double, ptr %d0_b_ptr, align 8
  %d5_b2 = load double, ptr %d5_b_ptr, align 8
  %bar_b2 = load i32, ptr @g_barriers, align 4
  %crit_b2 = load i32, ptr @g_crit_acquires, align 4
  %depth_b2 = load i32, ptr @g_crit_depth, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_blk, i32 2, i32 %cnt_b2, double %d0_b2, double %d5_b2, i32 %bar_b2, i32 %crit_b2, i32 %depth_b2)
  br label %zero6

zero6:
  %z6 = phi i64 [ 0, %run_b2 ], [ %z6_next, %zero6 ]
  %pz6 = getelementptr inbounds [6 x double], ptr %d_arr6, i64 0, i64 %z6
  store double 0.000000e+00, ptr %pz6, align 8
  %z6_next = add i64 %z6, 1
  %cz6 = icmp slt i64 %z6_next, 6
  br i1 %cz6, label %zero6, label %run_b1

run_b1:
  store i32 1, ptr @g_reduce_mode, align 4
  store i32 0, ptr @g_barriers, align 4
  store i32 0, ptr @g_crit_acquires, align 4
  store i32 0, ptr %cnt_box, align 4
  %ign3 = call double (...) @__enzyme_autodiff(ptr @blocking_mixed_reduce, i32 %dup_code, ptr %arr6, ptr %d_arr6, i32 %const_code, i32 6, i32 %const_code, i32 0, i32 %const_code, ptr %cnt_box)
  %cnt_b1 = load i32, ptr %cnt_box, align 4
  %d0_b1 = load double, ptr %d0_b_ptr, align 8
  %d5_b1 = load double, ptr %d5_b_ptr, align 8
  %bar_b1 = load i32, ptr @g_barriers, align 4
  %crit_b1 = load i32, ptr @g_crit_acquires, align 4
  %depth_b1 = load i32, ptr @g_crit_depth, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_blk, i32 1, i32 %cnt_b1, double %d0_b1, double %d5_b1, i32 %bar_b1, i32 %crit_b1, i32 %depth_b1)
  ret i32 0
}

; CHECK: @.enzyme_omp_reduction_lock.orig_lock = internal global [8 x i32] zeroinitializer, align 8

; CHECK-LABEL: define internal void @augmented_outlined_plus_minus(
; CHECK: call i32 @__kmpc_reduce_nowait(ptr @loc_red, i32 %{{.+}}, i32 2, i64 16, ptr null, ptr null, ptr @orig_lock)

; CHECK-LABEL: define internal void @diffeoutlined_plus_minus(
; CHECK: call void @__kmpc_end_critical(ptr @loc_red, i32 %{{.+}}, ptr @.enzyme_omp_reduction_lock.orig_lock)
; CHECK: call void @__kmpc_critical(ptr @loc_red, i32 %{{.+}}, ptr @.enzyme_omp_reduction_lock.orig_lock)

; CHECK-LABEL: define internal void @augmented_outlined_blocking_mixed(
; CHECK: call i32 @__kmpc_reduce(ptr @loc_red, i32 %{{.+}}, i32 2, i64 16, ptr null, ptr null, ptr @orig_lock)

; CHECK-LABEL: define internal void @diffeoutlined_blocking_mixed(
; CHECK: call void @__kmpc_end_critical(ptr @loc_red, i32 %{{.+}}, ptr @.enzyme_omp_reduction_lock.orig_lock)
; CHECK: call void @__kmpc_barrier(ptr @loc_red, i32 %{{.+}})
; CHECK: call void @__kmpc_barrier(ptr @loc_red, i32 %{{.+}})
; CHECK: call void @__kmpc_critical(ptr @loc_red, i32 %{{.+}}, ptr @.enzyme_omp_reduction_lock.orig_lock)
; CHECK: call void @__kmpc_barrier(ptr @loc_red, i32 %{{.+}})
