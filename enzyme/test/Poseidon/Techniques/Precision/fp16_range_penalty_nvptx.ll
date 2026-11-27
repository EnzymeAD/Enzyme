; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-expansions=false -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-early-prune \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_half.csv -poseidon-print -S
; DEFINE: %{in} = -poseidon-profile-use=%S/../../Inputs/fp16_range/in_range
; DEFINE: %{out} = -poseidon-profile-use=%S/../../Inputs/fp16_range/out_of_range
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %{in} -poseidon-cache=%t/in16 -poseidon-cost-budget=-0.904 2>&1 | FileCheck --check-prefix=IN16 %s
; RUN: %{opt} %{out} -poseidon-cache=%t/out16 -poseidon-cost-budget=-0.904 2>&1 | FileCheck --check-prefix=OUT16 %s
; RUN: %{opt} %{in} -poseidon-cache=%t/in32 -poseidon-cost-budget=-0.9 2>&1 | FileCheck --check-prefix=IN32 %s
; RUN: %{opt} %{out} -poseidon-cache=%t/out32 -poseidon-cost-budget=-0.9 2>&1 | FileCheck --check-prefix=OUT32 %s

; IN16: Initial AccuracyCost: 6.432707e-14
; IN16: 4.192367e-05		-4.746240e+05		All FP64(0%) + FP32(100%)
; IN16: 3.358604e-01		-4.753920e+05		All FP64(0%) + FP16(100%)
; IN16: Minimum accuracy cost within budget: 3.358604e-01
; IN16-NEXT: Computation cost budget used: -9.049708e-01
; IN16: Applying solution for precision change All FP64(0%) + FP16(100%) (#2)
; IN16-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; IN16: fptrunc double %{{.+}} to half
; IN16: call half @llvm.fmuladd.f16(
; IN16: fpext half %{{.+}} to double

; OUT16: Initial AccuracyCost: 2.994823e-02
; OUT16: 2.254356e+07		-4.746240e+05		All FP64(0%) + FP32(100%)
; OUT16-NOT: FP16(100%)
; OUT16-NOT: NOT caching the DP table
; OUT16: [poseidon] No solution found within the computation cost budget
; OUT16-NOT: Applying solution
; OUT16-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; OUT16-NOT: half
; OUT16: ret void

; IN32: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; IN32-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; IN32-NOT: half
; IN32: fptrunc double %{{.+}} to float

; OUT32: Minimum accuracy cost within budget: 2.254356e+07
; OUT32: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; OUT32-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; OUT32-NOT: half
; OUT32: fptrunc double %{{.+}} to float

; One kernel, two profiles: x in [0.25, 0.75] (in_range) and x scaled by 1000
; (out_of_range, past FP16's 65504). The FP16 candidate costs the same under
; both, but overflows on the out_of_range samples, so the DP never selects it
; there: at a budget only FP16 reaches, in_range applies it and out_of_range
; nothing; at a budget FP32 reaches, both take FP32.
; Module of fp16_range.cu (an fms chain) via clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on; profiles from fms_kernel<<<4, 64>>>
; at both scales on an sm_120 GPU; the cost model prices half at half of float.

source_filename = "fp16_range/fp16_range.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [25 x i8] c"fp16_range/fp16_range.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z10fms_kernelPKdS0_Pd, ptr @.str, ptr @.str1, i32 6, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z10fms_kernelPKdS0_Pd(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
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
  %20 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %19, ptr %20, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
