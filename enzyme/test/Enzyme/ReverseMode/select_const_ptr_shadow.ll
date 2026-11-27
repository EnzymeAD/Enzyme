; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli -

@rodata_const = internal constant double 7.000000e+00, align 8
@enzyme_const = external global i32
@enzyme_dup = external global i32
@enzyme_out = external global i32

declare void @abort()
declare double @llvm.fabs.f64(double)
declare void @__enzyme_autodiff(ptr, ...)
declare double @__enzyme_autodiff_ret(ptr, ...)

define internal double @load_select_phi(ptr nocapture readonly %types, ptr nocapture readonly %val, ptr nocapture readonly %p, i1 %use_rodata) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum.next, %loop ]
  %type_ptr = getelementptr inbounds i32, ptr %types, i64 %iv
  %type_val = load i32, ptr %type_ptr, align 4
  %is_const = icmp eq i32 %type_val, 0
  %val_ptr = getelementptr inbounds double, ptr %val, i64 %iv
  %iv.plus1 = add nuw nsw i64 %iv, 1
  %p_idx = urem i64 %iv.plus1, 5
  %p_ptr = getelementptr inbounds double, ptr %p, i64 %p_idx
  %chosen_ptr = select i1 %is_const, ptr %val_ptr, ptr %p_ptr
  %loaded = load double, ptr %chosen_ptr, align 8
  %sum.next = fadd double %sum, %loaded
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, 5
  br i1 %done, label %after_loop, label %loop

after_loop:
  br i1 %use_rodata, label %from_rodata, label %from_active

from_rodata:
  br label %merge_phi

from_active:
  %p0 = getelementptr inbounds double, ptr %p, i64 0
  br label %merge_phi

merge_phi:
  %phi_ptr = phi ptr [ @rodata_const, %from_rodata ], [ %p0, %from_active ]
  %tail = load double, ptr %phi_ptr, align 8
  %total = fadd double %sum.next, %tail
  ret double %total
}

define internal void @store_select_callee(i1 %cond, ptr nocapture %const_buf, ptr nocapture %active_buf, double %x) #0 {
entry:
  %scaled = fmul double %x, 3.000000e+00
  %dst = select i1 %cond, ptr %const_buf, ptr %active_buf
  store double %scaled, ptr %dst, align 8
  ret void
}

define internal double @store_select_caller(i1 %cond, ptr nocapture %const_buf, ptr nocapture %active_buf, double %x) {
entry:
  call void @store_select_callee(i1 %cond, ptr %const_buf, ptr %active_buf, double %x)
  %loaded = load double, ptr %active_buf, align 8
  ret double %loaded
}

define internal void @assert_near(double %actual, double %expected) {
entry:
  %diff = fsub double %actual, %expected
  %abs = call double @llvm.fabs.f64(double %diff)
  %bad = fcmp ogt double %abs, 1.000000e-12
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret void
}

define i32 @main() {
entry:
  %types = alloca [5 x i32], align 16
  %val = alloca [5 x double], align 16
  %p = alloca [5 x double], align 16
  %dp = alloca [5 x double], align 16
  %const_buf = alloca double, align 8
  %active_buf = alloca double, align 8
  %d_active_buf = alloca double, align 8

  %t0 = getelementptr inbounds [5 x i32], ptr %types, i64 0, i64 0
  %t1 = getelementptr inbounds [5 x i32], ptr %types, i64 0, i64 1
  %t2 = getelementptr inbounds [5 x i32], ptr %types, i64 0, i64 2
  %t3 = getelementptr inbounds [5 x i32], ptr %types, i64 0, i64 3
  %t4 = getelementptr inbounds [5 x i32], ptr %types, i64 0, i64 4
  store i32 1, ptr %t0, align 4
  store i32 1, ptr %t1, align 4
  store i32 0, ptr %t2, align 4
  store i32 1, ptr %t3, align 4
  store i32 1, ptr %t4, align 4

  br label %init

init:
  %i = phi i64 [ 0, %entry ], [ %i.next, %init ]
  %vp = getelementptr inbounds [5 x double], ptr %val, i64 0, i64 %i
  %pp = getelementptr inbounds [5 x double], ptr %p, i64 0, i64 %i
  %dpp = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 %i
  store double 0.000000e+00, ptr %vp, align 8
  %i.plus1 = add nuw nsw i64 %i, 1
  %pval = uitofp i64 %i.plus1 to double
  store double %pval, ptr %pp, align 8
  store double 0.000000e+00, ptr %dpp, align 8
  %i.next = add nuw nsw i64 %i, 1
  %init.done = icmp eq i64 %i.next, 5
  br i1 %init.done, label %run, label %init

run:
  %c_flag = load i32, ptr @enzyme_const, align 4
  %d_flag = load i32, ptr @enzyme_dup, align 4
  %a_flag = load i32, ptr @enzyme_out, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @load_select_phi, i32 %c_flag, ptr %types, i32 %c_flag, ptr %val, i32 %d_flag, ptr %p, ptr %dp, i32 %c_flag, i1 true)

  %v2_ptr = getelementptr inbounds [5 x double], ptr %val, i64 0, i64 2
  %v2 = load double, ptr %v2_ptr, align 8
  call void @assert_near(double %v2, double 0.000000e+00)

  %dp0_ptr = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 0
  %dp1_ptr = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 1
  %dp2_ptr = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 2
  %dp3_ptr = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 3
  %dp4_ptr = getelementptr inbounds [5 x double], ptr %dp, i64 0, i64 4
  %dp0 = load double, ptr %dp0_ptr, align 8
  %dp1 = load double, ptr %dp1_ptr, align 8
  %dp2 = load double, ptr %dp2_ptr, align 8
  %dp3 = load double, ptr %dp3_ptr, align 8
  %dp4 = load double, ptr %dp4_ptr, align 8
  call void @assert_near(double %dp0, double 1.000000e+00)
  call void @assert_near(double %dp1, double 1.000000e+00)
  call void @assert_near(double %dp2, double 1.000000e+00)
  call void @assert_near(double %dp3, double 0.000000e+00)
  call void @assert_near(double %dp4, double 1.000000e+00)

  store double 9.000000e+00, ptr %const_buf, align 8
  store double 2.000000e+00, ptr %active_buf, align 8
  store double 0.000000e+00, ptr %d_active_buf, align 8
  %dx_const = call double (ptr, ...) @__enzyme_autodiff_ret(ptr @store_select_caller, i32 %c_flag, i1 true, i32 %c_flag, ptr %const_buf, i32 %d_flag, ptr %active_buf, ptr %d_active_buf, i32 %a_flag, double 4.000000e+00)
  %cb_after = load double, ptr %const_buf, align 8
  call void @assert_near(double %cb_after, double 1.200000e+01)
  call void @assert_near(double %dx_const, double 0.000000e+00)

  store double 9.000000e+00, ptr %const_buf, align 8
  store double 2.000000e+00, ptr %active_buf, align 8
  store double 0.000000e+00, ptr %d_active_buf, align 8
  %dx_act = call double (ptr, ...) @__enzyme_autodiff_ret(ptr @store_select_caller, i32 %c_flag, i1 false, i32 %c_flag, ptr %const_buf, i32 %d_flag, ptr %active_buf, ptr %d_active_buf, i32 %a_flag, double 4.000000e+00)
  %cb_untouched = load double, ptr %const_buf, align 8
  call void @assert_near(double %cb_untouched, double 9.000000e+00)
  call void @assert_near(double %dx_act, double 3.000000e+00)

  ret i32 0
}

attributes #0 = { noinline }

; CHECK-LABEL: define internal void @diffeload_select_phi(
; CHECK: icmp ne ptr
; CHECK: icmp ne ptr

; CHECK-LABEL: define internal { double } @diffestore_select_callee(
; CHECK: icmp ne ptr
