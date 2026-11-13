; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O3>" | %lli -

%struct.CentroidVolume = type { [2 x float], [2 x [3 x float]] }
%struct.FlaggedValue = type { i8, double }
%struct.WrappedFlagged = type { %struct.FlaggedValue, [2 x double] }

define %struct.CentroidVolume @compute_pre_work(ptr %op, i64 %index) noinline {
entry:
  %buffer = alloca %struct.CentroidVolume, align 4
  %cond = icmp eq i64 %index, 0
  br i1 %cond, label %active, label %done

active:
  %ptr = load ptr, ptr %op, align 8
  %val = load float, ptr %ptr, align 4
  %scaled = fmul float %val, 2.000000e+00
  %v0 = getelementptr inbounds %struct.CentroidVolume, ptr %buffer, i32 0, i32 0, i32 0
  store float %scaled, ptr %v0, align 4
  br label %done

done:
  %result = load %struct.CentroidVolume, ptr %buffer, align 4
  ret %struct.CentroidVolume %result
}

define void @accumulate_post_work(ptr %op, ptr %cv) noinline {
entry:
  %ptr = load ptr, ptr %op, align 8
  %v0_ptr = getelementptr inbounds %struct.CentroidVolume, ptr %cv, i32 0, i32 0, i32 0
  %v0 = load float, ptr %v0_ptr, align 4
  %weighted = fmul float %v0, 3.000000e+00
  %old = load float, ptr %ptr, align 4
  %sum = fadd float %old, %weighted
  store float %sum, ptr %ptr, align 4
  ret void
}

define void @kernel_extract(ptr %op, i64 %index) {
entry:
  %local = alloca %struct.CentroidVolume, align 4
  %cv = call %struct.CentroidVolume @compute_pre_work(ptr %op, i64 %index)
  %arr0 = extractvalue %struct.CentroidVolume %cv, 0
  %elem0 = extractvalue [2 x float] %arr0, 0
  %dst0 = getelementptr inbounds %struct.CentroidVolume, ptr %local, i32 0, i32 0, i32 0
  store float %elem0, ptr %dst0, align 4
  %arr1 = extractvalue %struct.CentroidVolume %cv, 1
  %elem1 = extractvalue [2 x [3 x float]] %arr1, 0, 0
  %dst1 = getelementptr inbounds %struct.CentroidVolume, ptr %local, i32 0, i32 1, i32 0, i32 0
  store float %elem1, ptr %dst1, align 4
  call void @accumulate_post_work(ptr %op, ptr %local)
  ret void
}

define %struct.WrappedFlagged @make_flagged(double %x, i64 %index) noinline {
entry:
  %buffer = alloca %struct.WrappedFlagged, align 8
  %cond = icmp eq i64 %index, 0
  br i1 %cond, label %active, label %done

active:
  %v1 = fmul double %x, 4.000000e+00
  %v2 = fmul double %x, 6.000000e+00
  %p1 = getelementptr inbounds %struct.WrappedFlagged, ptr %buffer, i32 0, i32 0, i32 1
  store double %v1, ptr %p1, align 8
  %p2 = getelementptr inbounds %struct.WrappedFlagged, ptr %buffer, i32 0, i32 1, i32 1
  store double %v2, ptr %p2, align 8
  br label %done

done:
  %result = load %struct.WrappedFlagged, ptr %buffer, align 8
  ret %struct.WrappedFlagged %result
}

define double @flagged_roundtrip(double %x, i64 %index) {
entry:
  %raw = call %struct.WrappedFlagged @make_flagged(double %x, i64 %index)
  %fv = extractvalue %struct.WrappedFlagged %raw, 0
  %arr = extractvalue %struct.WrappedFlagged %raw, 1
  %rebuilt0 = insertvalue %struct.WrappedFlagged zeroinitializer, %struct.FlaggedValue %fv, 0
  %rebuilt1 = insertvalue %struct.WrappedFlagged %rebuilt0, [2 x double] %arr, 1
  %fv_out = extractvalue %struct.WrappedFlagged %rebuilt1, 0
  %d0 = extractvalue %struct.FlaggedValue %fv_out, 1
  %d1 = extractvalue %struct.WrappedFlagged %rebuilt1, 1, 1
  %sum = fadd double %d0, %d1
  ret double %sum
}

declare void @__enzyme_autodiff(...)
declare double @__enzyme_autodiff_f64(...)

define float @run_kernel_grad(float %input) {
entry:
  %x = alloca float, align 4
  %dx = alloca float, align 4
  %op = alloca ptr, align 8
  %dop = alloca ptr, align 8
  store float %input, ptr %x, align 4
  store float 1.000000e+00, ptr %dx, align 4
  store ptr %x, ptr %op, align 8
  store ptr %dx, ptr %dop, align 8
  call void (...) @__enzyme_autodiff(ptr @kernel_extract, metadata !"enzyme_dup", ptr %op, ptr %dop, i64 0)
  %grad = load float, ptr %dx, align 4
  ret float %grad
}

define double @run_flagged_grad(double %x) {
entry:
  %grad = call double (...) @__enzyme_autodiff_f64(ptr @flagged_roundtrip, double %x, i64 0)
  ret double %grad
}

define i32 @main() {
entry:
  %g1 = call float @run_kernel_grad(float 3.000000e+00)
  %err1 = fsub float %g1, 7.000000e+00
  %neg1 = fsub float 0.000000e+00, %err1
  %isneg1 = fcmp olt float %err1, 0.000000e+00
  %abs1 = select i1 %isneg1, float %neg1, float %err1
  %ok1 = fcmp olt float %abs1, 9.765625e-04

  %g2 = call double @run_flagged_grad(double 2.500000e+00)
  %err2 = fsub double %g2, 1.000000e+01
  %neg2 = fsub double 0.000000e+00, %err2
  %isneg2 = fcmp olt double %err2, 0.000000e+00
  %abs2 = select i1 %isneg2, double %neg2, double %err2
  %ok2 = fcmp olt double %abs2, 1.000000e-09

  %both = and i1 %ok1, %ok2
  %ret = select i1 %both, i32 0, i32 1
  ret i32 %ret
}

; CHECK-LABEL: define internal void @diffekernel_extract(
; CHECK: call void @diffecompute_pre_work(

; CHECK-LABEL: define internal { double } @diffeflagged_roundtrip(
; CHECK: call { double } @diffemake_flagged(
