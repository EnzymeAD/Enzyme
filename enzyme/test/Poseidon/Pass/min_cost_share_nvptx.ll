; REQUIRES: cuda-runtime
; From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
; DEFINE: %{use} = -poseidon-kernels=.*_kernel \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/min_cost_share \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 \
; DEFINE:   -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-1e-9 -poseidon-print
; RUN: rm -rf %t && mkdir %t %t/tmp
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-cache=%t/filter -S 2>&1 \
; RUN:   | FileCheck --implicit-check-not="Optimizing preprocess_cold_kernel_poseidon_body" %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-min-cost-share=0 -poseidon-cache=%t/nofilter -S 2>&1 \
; RUN:   | FileCheck --check-prefix=NOFILTER %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon "-poseidon-kernels=(" -poseidon-profile-use=%S/../Inputs/min_cost_share -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=BADREGEX %s
; RUN: not env TMPDIR=%t/tmp %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -c -x ir %s %loadClangPoseidon -mllvm "-poseidon-kernels=(" -mllvm -poseidon-profile-use=%S/../Inputs/min_cost_share -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -mllvm -poseidon-cache= -o %t/tmp/k.o 2>&1 | FileCheck --check-prefix=BADREGEX --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: ls %t/tmp | count 0

; Two kernels reached by -poseidon-kernels, not by the attribute. cold_kernel
; ran once where hot_kernel ran 768 times over seven profiled operations, so it
; is 0.026% of the profiled FP cost of the two and is left exactly as it was
; found. The second run, with the share set to 0, optimizes both.

; ModuleID = 'min_cost_share.cu'
source_filename = "min_cost_share.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define dso_local ptx_kernel void @hot_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8
  ret void
}

define dso_local ptx_kernel void @cold_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %14, ptr %15, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

; CHECK: [poseidon] Optimizing preprocess_hot_kernel_poseidon_body
; CHECK: [poseidon] cold_kernel is 0.02599% of the profiled FP cost of the kernels -poseidon-kernels named, below -poseidon-min-cost-share=0.01; left unoptimized
; CHECK-LABEL: define {{.*}} @hot_kernel(
; CHECK: call void @preprocess_hot_kernel_poseidon_body(
; CHECK-LABEL: define {{.*}} @cold_kernel(
; CHECK-NOT: _poseidon_body
; CHECK: call double @llvm.fmuladd.f64(

; NOFILTER-NOT: left unoptimized
; NOFILTER: [poseidon] Optimizing preprocess_hot_kernel_poseidon_body
; NOFILTER: [poseidon] Optimizing preprocess_cold_kernel_poseidon_body

; A -poseidon-kernels value that is no regular expression is an error of the
; compile, under opt and under clang, without clang's crash reporting.
; BADREGEX: error: Poseidon: -poseidon-kernels=( is not a valid regular expression: {{.+}}. Give a regular expression of kernel names, or 'all'.
