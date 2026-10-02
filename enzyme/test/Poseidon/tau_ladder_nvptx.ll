; RUN: rm -rf %t && mkdir -p %t
; RUN: sed "s/1e-07/1e-12/" %s > %t/mid.ll
; RUN: sed "s/1e-07/1e-18/" %s > %t/tight.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=LOOSE %s
; RUN: %opt %t/mid.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=MID %s
; RUN: %opt %t/tight.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=TIGHT %s
; REQUIRES: poseidon

; A site tolerance is a per-operation relative rounding level: the solve takes
; the CHEAPEST frontier point whose modelled error, put back on that scale,
; clears it. Same kernel and profile as annotated_kernel_nvptx.ll, solved three
; times against one cached DP table (so the second and third runs also read the
; baseline back out of the table rather than repricing it).
;
; The site's own FP64 body sits at A0/S = 2.16e-17, which is what makes the
; third arm infeasible rather than a silent no-op with a made-up number.

; ModuleID = 'tau_ladder.cu'
source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [19 x i8] c"poseidon;tau=1e-07\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [14 x i8] c"tau_ladder.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local ptx_kernel void @fms_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !9
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8, !tbaa !9
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8, !tbaa !9
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git a70419505471bd8240ef3451dcdd541f8676477c)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !7, i64 0}




; The annotation alone must not change the kernel: with no profile the site is
; left as written (no outline, no call), and with a profile whose solve applies
; nothing the outline is folded back. With a rewrite applied the materialized
; body is folded back the same way. In every case the kernel is one entry
; function with the arithmetic inline and no _poseidon_body symbol survives.

; LOOSE: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-07 S=1.187943e+04 A0=2.571104e-13 selected cost={{-[0-9]+}} accCost={{.*}} rel=1.428768e-08
; LOOSE: Applying solution for CS: All FP64(0%) + FP32(100%)
; LOOSE-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LOOSE: fptrunc double
; LOOSE-NOT: !poseidon.ds.join

; MID: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost={{-[0-9]+}} accCost={{.*}} rel=5.925412e-16
; MID: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; MID-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; MID: !poseidon.ds.join

; A tolerance below what any point on the frontier reaches is refused, and the
; site keeps the original body bit-for-bit.
; TIGHT: No solution found that meets accuracy tolerance 1.000000e-18!
; TIGHT: Best achievable relative accuracy in DP table: 2.164332e-17
; TIGHT: [poseidon] no rewrite applied for preprocess_fms_kernel_poseidon_body
; TIGHT-LABEL: define {{.*}} @fms_kernel(
; TIGHT: call double @llvm.fmuladd.f64
; TIGHT-NOT: !poseidon.ds.join
