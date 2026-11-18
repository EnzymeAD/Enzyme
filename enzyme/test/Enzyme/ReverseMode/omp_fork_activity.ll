; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: fork_activity_ok flag=12 res=300.000000 d0=6.000000 d3=24.000000

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str.loc = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@loc = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @.str.loc }, align 8
@.fmt = private unnamed_addr constant [45 x i8] c"fork_activity_ok flag=%d res=%f d0=%f d3=%f\0A\00", align 1

@g_tid = internal global i32 0, align 4

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

define void @__kmpc_fork_call(ptr %ident, i32 %argc, ptr %microtask, ...) noinline {
entry:
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
  %t_more = icmp slt i32 %tid, 4
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
  %p5 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 5
  %a5 = load ptr, ptr %p5, align 8
  %p6 = getelementptr inbounds [8 x ptr], ptr %args_arr, i32 0, i32 6
  %a6 = load ptr, ptr %p6, align 8
  switch i32 %argc, label %bad_argc [
    i32 4, label %call4
    i32 6, label %call6
    i32 7, label %call7
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

t_cont:
  %tid_next = add i32 %tid, 1
  br label %t_loop

t_done:
  store i32 0, ptr @g_tid, align 4
  ret void
}

define internal void @outlined_fork_activity(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture align 8 %sum_ptr, ptr align 8 %const_slot, ptr nocapture readonly align 8 %active_arr, ptr align 4 %flag_ptr) {
entry:
  %spill = alloca double, align 8
  %gtid = load i32, ptr %gtid_ptr, align 4
  %idx = sext i32 %gtid to i64
  %const_arr = load ptr, ptr %const_slot, align 8
  %c_ptr = getelementptr inbounds double, ptr %const_arr, i64 %idx
  %c_val = load double, ptr %c_ptr, align 8
  store double %c_val, ptr %spill, align 8
  %c_reloaded = load double, ptr %spill, align 8
  %gt5 = fcmp ogt double %c_reloaded, 5.000000e+00
  %shl_amt = and i32 %gtid, 7
  %mask_raw = shl i32 1, %shl_amt
  %mask = select i1 %gt5, i32 %mask_raw, i32 0
  %old_flag = atomicrmw add ptr %flag_ptr, i32 %mask monotonic, align 4
  %a_ptr = getelementptr inbounds double, ptr %active_arr, i64 %idx
  %a_val = load double, ptr %a_ptr, align 8
  %prod = fmul double %c_reloaded, %a_val
  %old_sum = atomicrmw fadd ptr %sum_ptr, double %prod monotonic, align 8
  ret void
}

define double @fork_activity_kernel(ptr %const_arr, ptr %active_arr, ptr %flags, ptr %out_sum) {
entry:
  %sum_slot = alloca double, align 8
  %const_slot = alloca ptr, align 8
  store double 0.000000e+00, ptr %sum_slot, align 8
  store ptr %const_arr, ptr %const_slot, align 8
  %flag_gep = getelementptr inbounds i32, ptr %flags, i64 1
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 4, ptr @outlined_fork_activity, ptr %sum_slot, ptr %const_slot, ptr %active_arr, ptr %flag_gep)
  %s = load double, ptr %sum_slot, align 8
  %scaled = fmul double 3.000000e+00, %s
  store double %scaled, ptr %out_sum, align 8
  ret double %scaled
}

define i32 @main() {
entry:
  %const_arr = alloca [4 x double], align 8
  %active_arr = alloca [4 x double], align 8
  %d_active_arr = alloca [4 x double], align 8
  %flags = alloca [2 x i32], align 4
  %out_sum = alloca double, align 8
  br label %init

init:
  %i = phi i64 [ 0, %entry ], [ %i_next, %init ]
  %pc = getelementptr inbounds [4 x double], ptr %const_arr, i64 0, i64 %i
  %pa = getelementptr inbounds [4 x double], ptr %active_arr, i64 0, i64 %i
  %pda = getelementptr inbounds [4 x double], ptr %d_active_arr, i64 0, i64 %i
  %ip1 = add i64 %i, 1
  %c_i = mul i64 %ip1, 2
  %a_i = sub i64 %c_i, 1
  %cv = sitofp i64 %c_i to double
  %av = sitofp i64 %a_i to double
  store double %cv, ptr %pc, align 8
  store double %av, ptr %pa, align 8
  store double 0.000000e+00, ptr %pda, align 8
  %i_next = add i64 %i, 1
  %cond = icmp slt i64 %i_next, 4
  br i1 %cond, label %init, label %run

run:
  %f0 = getelementptr inbounds [2 x i32], ptr %flags, i64 0, i64 0
  %f1 = getelementptr inbounds [2 x i32], ptr %flags, i64 0, i64 1
  store i32 0, ptr %f0, align 4
  store i32 0, ptr %f1, align 4
  store double 0.000000e+00, ptr %out_sum, align 8
  %ign = call double (...) @__enzyme_autodiff(ptr @fork_activity_kernel, metadata !"enzyme_const", ptr %const_arr, metadata !"enzyme_dup", ptr %active_arr, ptr %d_active_arr, metadata !"enzyme_const", ptr %flags, metadata !"enzyme_const", ptr %out_sum)
  %flag_val = load i32, ptr %f1, align 4
  %res_val = load double, ptr %out_sum, align 8
  %d0_ptr = getelementptr inbounds [4 x double], ptr %d_active_arr, i64 0, i64 0
  %d3_ptr = getelementptr inbounds [4 x double], ptr %d_active_arr, i64 0, i64 3
  %d0 = load double, ptr %d0_ptr, align 8
  %d3 = load double, ptr %d3_ptr, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt, i32 %flag_val, double %res_val, double %d0, double %d3)
  ret i32 0
}

; Only sum_slot and active_arr are active; const_slot and flag_gep stay unshadowed.
; CHECK-LABEL: define internal void @diffefork_activity_kernel(
; CHECK: %[[FLAG_GEP:.+]] = getelementptr inbounds i32, ptr %flags, i64 1
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 {{6|7}}, ptr @diffeoutlined_fork_activity, ptr %{{.+}}, ptr %{{.+}}, ptr %const_slot, ptr %active_arr, ptr %"active_arr'", ptr %[[FLAG_GEP]]

; CHECK-LABEL: define internal void @diffeoutlined_fork_activity(
; CHECK-SAME: ptr {{.*}}%sum_ptr, ptr {{.*}}%"sum_ptr'", ptr {{.*}}%const_slot, ptr {{.*}}%active_arr, ptr {{.*}}%"active_arr'", ptr {{.*}}%flag_ptr
; CHECK: atomicrmw fadd ptr %{{.+}}, double %{{.+}} monotonic
; CHECK: ret void
