; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli -

@enzyme_const = external global i32
@enzyme_dup = external global i32

declare void @abort()
declare double @llvm.fabs.f64(double)
declare double @__enzyme_fwddiff(ptr, ...)
declare void @__enzyme_fwddiff_void(ptr, ...)

define internal double @fwd_load_select(i1 %cond, ptr nocapture readonly %const_arr, ptr nocapture readonly %active_arr) {
entry:
  %chosen = select i1 %cond, ptr %const_arr, ptr %active_arr
  %loaded = load double, ptr %chosen, align 8
  ret double %loaded
}

define internal void @fwd_store_select(i1 %cond, ptr nocapture %const_out, ptr nocapture %active_out, double %x) {
entry:
  %scaled = fmul double %x, 2.000000e+00
  %dst = select i1 %cond, ptr %const_out, ptr %active_out
  store double %scaled, ptr %dst, align 8
  ret void
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
  %const_arr = alloca double, align 8
  %active_arr = alloca double, align 8
  %d_active_arr = alloca double, align 8
  %const_out = alloca double, align 8
  %active_out = alloca double, align 8
  %d_active_out = alloca double, align 8

  store double 3.000000e+01, ptr %const_arr, align 8
  store double 4.000000e+00, ptr %active_arr, align 8
  store double 5.000000e+00, ptr %d_active_arr, align 8

  %c_flag = load i32, ptr @enzyme_const, align 4
  %d_flag = load i32, ptr @enzyme_dup, align 4

  %dret_const = call double (ptr, ...) @__enzyme_fwddiff(ptr @fwd_load_select, i32 %c_flag, i1 true, i32 %c_flag, ptr %const_arr, i32 %d_flag, ptr %active_arr, ptr %d_active_arr)
  call void @assert_near(double %dret_const, double 0.000000e+00)

  %dret_act = call double (ptr, ...) @__enzyme_fwddiff(ptr @fwd_load_select, i32 %c_flag, i1 false, i32 %c_flag, ptr %const_arr, i32 %d_flag, ptr %active_arr, ptr %d_active_arr)
  call void @assert_near(double %dret_act, double 5.000000e+00)

  store double -1.000000e+00, ptr %const_out, align 8
  store double -1.000000e+00, ptr %active_out, align 8
  store double 7.000000e+00, ptr %d_active_out, align 8
  call void (ptr, ...) @__enzyme_fwddiff_void(ptr @fwd_store_select, i32 %c_flag, i1 true, i32 %c_flag, ptr %const_out, i32 %d_flag, ptr %active_out, ptr %d_active_out, i32 %d_flag, double 4.000000e+00, double 3.000000e+00)
  %co_val = load double, ptr %const_out, align 8
  %dao_untouched = load double, ptr %d_active_out, align 8
  call void @assert_near(double %co_val, double 8.000000e+00)
  call void @assert_near(double %dao_untouched, double 7.000000e+00)

  store double 9.000000e+00, ptr %const_out, align 8
  store double -1.000000e+00, ptr %active_out, align 8
  store double 7.000000e+00, ptr %d_active_out, align 8
  call void (ptr, ...) @__enzyme_fwddiff_void(ptr @fwd_store_select, i32 %c_flag, i1 false, i32 %c_flag, ptr %const_out, i32 %d_flag, ptr %active_out, ptr %d_active_out, i32 %d_flag, double 4.000000e+00, double 3.000000e+00)
  %co_still = load double, ptr %const_out, align 8
  %ao_val = load double, ptr %active_out, align 8
  %dao_val = load double, ptr %d_active_out, align 8
  call void @assert_near(double %co_still, double 9.000000e+00)
  call void @assert_near(double %ao_val, double 8.000000e+00)
  call void @assert_near(double %dao_val, double 6.000000e+00)

  ret i32 0
}

; CHECK-LABEL: define internal double @fwddiffefwd_load_select(
; CHECK: %[[CMP_LOAD:.+]] = icmp ne ptr %chosen, %"chosen'ipse"
; CHECK: select i1 %[[CMP_LOAD]], double %{{.+}}, double 0.000000e+00

; CHECK-LABEL: define internal void @fwddiffefwd_store_select(
; CHECK: %[[SINK:.+]] = alloca [8 x i8], align 8
; CHECK: %[[CMP_STORE:.+]] = icmp eq ptr %"dst'ipse", %dst
; CHECK: %[[GUARDED_DST:.+]] = select i1 %[[CMP_STORE]], ptr %[[SINK]], ptr %"dst'ipse"
; CHECK: store double %{{.+}}, ptr %[[GUARDED_DST]], align 8
