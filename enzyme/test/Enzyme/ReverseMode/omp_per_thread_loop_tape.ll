; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: d0=32.000000 d1=108.000000 d2=500.000000

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str.loc = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@loc = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @.str.loc }, align 8
@.fmt = private unnamed_addr constant [19 x i8] c"d0=%f d1=%f d2=%f\0A\00", align 1

@g_tid = internal global i32 0, align 4
@g_nthreads = internal global i32 3, align 4

declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr, ...)
declare void @abort()
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define i64 @omp_get_max_threads() noinline {
entry:
  %n = load i32, ptr @g_nthreads, align 4
  %n64 = zext i32 %n to i64
  ret i64 %n64
}

define i64 @omp_get_thread_num() noinline {
entry:
  %t = load i32, ptr @g_tid, align 4
  %t64 = zext i32 %t to i64
  ret i64 %t64
}

define void @__kmpc_fork_call(ptr %ident, i32 %argc, ptr %microtask, ...) noinline {
entry:
  %args_arr = alloca [6 x ptr], align 8
  %ap = alloca [64 x i8], align 16
  call void @llvm.va_start(ptr %ap)
  br label %va_loop

va_loop:
  %idx = phi i32 [ 0, %entry ], [ %idx_next, %va_body ]
  %more_va = icmp slt i32 %idx, %argc
  br i1 %more_va, label %va_body, label %va_done

va_body:
  %val = va_arg ptr %ap, ptr
  %slot = getelementptr inbounds [6 x ptr], ptr %args_arr, i32 0, i32 %idx
  store ptr %val, ptr %slot, align 8
  %idx_next = add i32 %idx, 1
  br label %va_loop

va_done:
  call void @llvm.va_end(ptr %ap)
  %gtid_slot = alloca i32, align 4
  %btid_slot = alloca i32, align 4
  %nthreads = load i32, ptr @g_nthreads, align 4
  br label %t_loop

t_loop:
  %tid = phi i32 [ 0, %va_done ], [ %tid_next, %t_cont ]
  %more = icmp slt i32 %tid, %nthreads
  br i1 %more, label %t_body, label %t_done

t_body:
  store i32 %tid, ptr @g_tid, align 4
  store i32 %tid, ptr %gtid_slot, align 4
  store i32 %tid, ptr %btid_slot, align 4
  %p0 = getelementptr inbounds [6 x ptr], ptr %args_arr, i32 0, i32 0
  %a0 = load ptr, ptr %p0, align 8
  %p1 = getelementptr inbounds [6 x ptr], ptr %args_arr, i32 0, i32 1
  %a1 = load ptr, ptr %p1, align 8
  %p2 = getelementptr inbounds [6 x ptr], ptr %args_arr, i32 0, i32 2
  %a2 = load ptr, ptr %p2, align 8
  switch i32 %argc, label %bad_argc [
    i32 1, label %call1
    i32 2, label %call2
    i32 3, label %call3
  ]

bad_argc:
  call void @abort()
  unreachable

call1:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0)
  br label %t_cont

call2:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1)
  br label %t_cont

call3:
  call void %microtask(ptr %gtid_slot, ptr %btid_slot, ptr %a0, ptr %a1, ptr %a2)
  br label %t_cont

t_cont:
  %tid_next = add i32 %tid, 1
  br label %t_loop

t_done:
  store i32 0, ptr @g_tid, align 4
  ret void
}

define internal void @outlined_per_thread_loop(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture align 8 %arr) {
entry:
  %tid32 = load i32, ptr %gtid_ptr, align 4
  %tid64 = sext i32 %tid32 to i64
  %slot = getelementptr inbounds double, ptr %arr, i64 %tid64
  br label %loop

loop:
  %step = phi i32 [ 0, %entry ], [ %step_next, %loop ]
  %cur = load double, ptr %slot, align 8
  %sq = fmul double %cur, %cur
  store double %sq, ptr %slot, align 8
  %step_next = add nuw nsw i32 %step, 1
  %cont = icmp slt i32 %step_next, 2
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

define double @parallel_fourth_power_sum(ptr %arr) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 1, ptr @outlined_per_thread_loop, ptr %arr)
  %p0 = getelementptr inbounds double, ptr %arr, i64 0
  %p1 = getelementptr inbounds double, ptr %arr, i64 1
  %p2 = getelementptr inbounds double, ptr %arr, i64 2
  %v0 = load double, ptr %p0, align 8
  %v1 = load double, ptr %p1, align 8
  %v2 = load double, ptr %p2, align 8
  %s01 = fadd double %v0, %v1
  %s012 = fadd double %s01, %v2
  ret double %s012
}

define i32 @main() {
entry:
  %arr = alloca [3 x double], align 8
  %d_arr = alloca [3 x double], align 8
  %a0 = getelementptr inbounds [3 x double], ptr %arr, i64 0, i64 0
  %a1 = getelementptr inbounds [3 x double], ptr %arr, i64 0, i64 1
  %a2 = getelementptr inbounds [3 x double], ptr %arr, i64 0, i64 2
  store double 2.000000e+00, ptr %a0, align 8
  store double 3.000000e+00, ptr %a1, align 8
  store double 5.000000e+00, ptr %a2, align 8

  %da0 = getelementptr inbounds [3 x double], ptr %d_arr, i64 0, i64 0
  %da1 = getelementptr inbounds [3 x double], ptr %d_arr, i64 0, i64 1
  %da2 = getelementptr inbounds [3 x double], ptr %d_arr, i64 0, i64 2
  store double 0.000000e+00, ptr %da0, align 8
  store double 0.000000e+00, ptr %da1, align 8
  store double 0.000000e+00, ptr %da2, align 8

  %ign = call double (ptr, ...) @__enzyme_autodiff(ptr @parallel_fourth_power_sum, metadata !"enzyme_dup", ptr %arr, ptr %d_arr)
  %d0 = load double, ptr %da0, align 8
  %d1 = load double, ptr %da1, align 8
  %d2 = load double, ptr %da2, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d0, double %d1, double %d2)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffeparallel_fourth_power_sum(
; CHECK: %[[MAXT:.+]] = call i64 @omp_get_max_threads()

; CHECK-LABEL: define internal void @augmented_outlined_per_thread_loop(
; CHECK: %[[TID_F:.+]] = call i64 @omp_get_thread_num()
; CHECK: getelementptr inbounds ptr, ptr %{{.+}}, i64 %[[TID_F]]

; CHECK-LABEL: define internal void @diffeoutlined_per_thread_loop(
; CHECK: %[[TID_R:.+]] = call i64 @omp_get_thread_num()
; CHECK: getelementptr inbounds ptr, ptr %{{.+}}, i64 %[[TID_R]]
