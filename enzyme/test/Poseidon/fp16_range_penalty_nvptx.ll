; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_half_x1e6.csv -poseidon-print -S -poseidon-profile-use=%S/Inputs/fp16_range/in_range -poseidon-cache=%t/in16 -poseidon-comp-cost-budget=-46600000 2>&1 | FileCheck --check-prefix=IN16 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_half_x1e6.csv -poseidon-print -S -poseidon-profile-use=%S/Inputs/fp16_range/out_of_range -poseidon-cache=%t/out16 -poseidon-comp-cost-budget=-46600000 2>&1 | FileCheck --check-prefix=OUT16 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_half_x1e6.csv -poseidon-print -S -poseidon-profile-use=%S/Inputs/fp16_range/in_range -poseidon-cache=%t/in32 -poseidon-comp-cost-budget=-1000 2>&1 | FileCheck --check-prefix=IN32 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_half_x1e6.csv -poseidon-print -S -poseidon-profile-use=%S/Inputs/fp16_range/out_of_range -poseidon-cache=%t/out32 -poseidon-comp-cost-budget=-1000 2>&1 | FileCheck --check-prefix=OUT32 %s
; REQUIRES: poseidon
; FP16 range pricing of an elementwise precision-tuning candidate. One kernel, two profiles of it: x in
; [0.25, 0.75] (in_range: every value of the FP subgraph lies inside FP16's range) and x scaled by 1000
; (out_of_range: t*t reaches 2.2e6 and the output 5e12, past FP16's 65504). The FP16 candidate is the
; same and costs the same under both; simulated in FP16 the out_of_range samples overflow, so its
; accuracy cost is non-finite and the DP never selects it: at a budget only FP16 reaches the in_range
; solve applies it and the out_of_range solve applies nothing; at a budget FP32 reaches, both take FP32.
; IR: Inputs/fp16_range/fp16_range.cu via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -I<enzyme>/include -S -emit-llvm. Profiles: the same source through poseidon-clang++
; -poseidon-profile-generate, run with argument 1 (in_range) and 1000 (out_of_range). Cost model:
; Inputs/cm_gpu_fixture_half_x1e6.csv (synthetic half rows, see its header).

; ModuleID = 'fp16_range/fp16_range.cu'
source_filename = "fp16_range/fp16_range.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [25 x i8] c"fp16_range/fp16_range.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z10fms_kernelPKdS0_Pd, ptr @.str, ptr @.str1, i32 6, ptr null }], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z10fms_kernelPKdS0_Pd(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8, !tbaa !10
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %19, ptr %20, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git e56c2cefc3e7978de4a2d799fa3abec77d75af21)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}

; IN16: Initial AccuracyCost: 6.432707e-14
; IN16: 4.192367e-05		-46561023		All FP64(0%) + FP32(100%)
; IN16: 3.358604e-01		-46621819		All FP64(0%) + FP16(100%)
; IN16: Minimum accuracy cost within budget: 3.358604e-01
; IN16-NEXT: Computation cost budget used: -46621819
; IN16: Applying solution for CS: All FP64(0%) + FP16(100%) (#2)
; IN16-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; IN16: fptrunc double %{{.+}} to half
; IN16: call half @llvm.fmuladd.f16(
; IN16: fpext half %{{.+}} to double

; OUT16: Initial AccuracyCost: 6.746930e+01
; OUT16: 5.078758e+10		-46561023		All FP64(0%) + FP32(100%)
; OUT16-NOT: FP16(100%)
; OUT16-NOT: NOT caching the DP table
; OUT16: No solution found within the computation cost budget!
; OUT16-NOT: Applying solution
; OUT16-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; OUT16-NOT: half
; OUT16: ret void

; IN32: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; IN32-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; IN32-NOT: half
; IN32: fptrunc double %{{.+}} to float

; OUT32: Minimum accuracy cost within budget: 5.078758e+10
; OUT32: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; OUT32-LABEL: define {{.*}} @_Z10fms_kernelPKdS0_Pd(
; OUT32-NOT: half
; OUT32: fptrunc double %{{.+}} to float
