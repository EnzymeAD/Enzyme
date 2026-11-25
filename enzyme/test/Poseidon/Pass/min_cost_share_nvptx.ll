; REQUIRES: cuda-toolkit, nvptx-registered-target
; From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
; DEFINE: %{use} = -poseidon-kernels=.*_kernel \
; DEFINE:   -poseidon-profile-use=%t/prof -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-1e-9 -poseidon-print
; RUN: rm -rf %t && mkdir %t %t/tmp %t/stale %t/prof
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/prof/hot_kernel.fpprofile
; RUN: cp %S/Inputs/min_cost_share/cold_kernel.fpprofile %t/prof/
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-cache=%t/filter -poseidon-explain=%t/filter/ex -S 2>&1 \
; RUN:   | tee %t/filter.log | FileCheck --implicit-check-not="Optimizing preprocess_cold_kernel_poseidon_body" %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/filter.log %t/filter/ex | FileCheck --check-prefix=JSON %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-min-cost-share=0 -poseidon-cache=%t/nofilter -S 2>&1 \
; RUN:   | FileCheck --check-prefix=NOFILTER %s
; RUN: cp %t/prof/*.fpprofile %t/stale/
; RUN: sed -i 's/^CanonicalHash = .*/CanonicalHash = 0123456789abcdef/' %t/stale/cold_kernel.fpprofile
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-profile-use=%t/stale -poseidon-early-prune -poseidon-cache=%t/stale -S -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=STALE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon "-poseidon-kernels=(" -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=BADREGEX %s
; RUN: not env TMPDIR=%t/tmp %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -c -x ir %s %loadClangPoseidon -mllvm "-poseidon-kernels=(" -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-early-prune -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -mllvm -poseidon-cache= -o %t/tmp/k.o 2>&1 | FileCheck --check-prefix=BADREGEX --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: ls %t/tmp | count 0

; Two kernels reached by -poseidon-kernels, not by the attribute. cold_kernel
; ran once where hot_kernel ran 768 times over seven profiled operations, so it
; is 0.026% of the profiled FP cost of the two and is left unoptimized, with
; the computation it was found with. The second run, with the share set to 0,
; hands both to the optimizer.
; JSON: cold_kernel_poseidon_body: below_cost_share; the site is 0.02{{[0-9]+}}% of the profiled FP cost of the kernels -poseidon-kernels named, below -poseidon-min-cost-share=0.01
; JSON-NEXT: preprocess_hot_kernel_poseidon_body: {{[a-z_]+}}

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


; What poseidon.h defines in device code under POSEIDON_DEVICE_MATH.
define linkonce_odr void @__poseidon_device_math_defined() {
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

; CHECK: [poseidon] Optimizing preprocess_hot_kernel_poseidon_body with relative error tolerance: 0.000000e+00{{$}}
; CHECK: [poseidon] cold_kernel is 0.02599% of the profiled FP cost of the kernels -poseidon-kernels named, below -poseidon-min-cost-share=0.01; left unoptimized{{$}}
; CHECK-LABEL: define {{.*}} @hot_kernel(
; CHECK-SAME: ptr {{.*}}[[HA:%[0-9]+]], ptr {{.*}}[[HB:%[0-9]+]], ptr {{.*}}[[HC:%[0-9]+]])
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @preprocess_hot_kernel_poseidon_body(ptr [[HA]], ptr [[HB]], ptr [[HC]])
; CHECK-NEXT: ret void
; CHECK-LABEL: define {{.*}} @cold_kernel(
; CHECK-SAME: ptr {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[B:%[0-9]+]], ptr {{.*}}[[C:%[0-9]+]])
; CHECK-NOT: _poseidon_body
; CHECK: [[R:%[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double [[AV:%[0-9]+]], double %{{[0-9]+}}, double [[AV]])
; CHECK-NEXT: [[CP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[C]], i64 %{{[0-9]+}}{{$}}
; CHECK-NEXT: store double [[R]], ptr [[CP]], align 8
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; NOFILTER-NOT: left unoptimized
; NOFILTER: [poseidon] Optimizing preprocess_hot_kernel_poseidon_body with relative error tolerance: 0.000000e+00{{$}}
; NOFILTER: [poseidon] Optimizing preprocess_cold_kernel_poseidon_body with relative error tolerance: 0.000000e+00{{$}}

; The share of a kernel is read from its profile, which has to be of the
; canonical form the compile numbers.
; STALE: {{^}}error: <unknown>:0:0: in function preprocess_cold_kernel_poseidon_body void (ptr, ptr, ptr): Poseidon: the profile of preprocess_cold_kernel_poseidon_body was recorded against a different canonical form (profile CanonicalHash 0123456789abcdef, this compile fea337c617ef94c9); re-run profile generation{{$}}

; A -poseidon-kernels value that is no regular expression is an error of the
; compile, under opt and under clang, without clang's crash reporting.
; BADREGEX: {{^}}error: Poseidon: -poseidon-kernels=( is not a valid regular expression (parentheses not balanced); give a regular expression of kernel names, or 'all'{{$}}
