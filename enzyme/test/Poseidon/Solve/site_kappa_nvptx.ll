; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1e-9 -poseidon-print -S
; Two sites whose profiles carry a condition number (Kappa 1000 on the product,
; 0.001 on the scalar subgraph): each candidate's accuracy cost is scaled by
; Kappa over the site's profiled sensitivity.
; RUN: rm -rf %t && mkdir -p %t/kappa %t/nokappa %t/diverged %t/noise
; RUN: (head -n 3 %S/../Inputs/matmul_site_profile/matmul_body.fpprofile && printf 'SiteId = 0\nKappa = 1000\nKappaMark = ok\n' && tail -n +4 %S/../Inputs/matmul_site_profile/matmul_body.fpprofile) > %t/kappa/matmul_body.fpprofile
; RUN: (head -n 3 %S/../Inputs/expansion_two_components/fms_body.fpprofile && printf 'SiteId = 1\nKappa = 0.001\nKappaMark = ok\n' && tail -n +4 %S/../Inputs/expansion_two_components/fms_body.fpprofile) > %t/kappa/fms_body.fpprofile
; RUN: cp %t/kappa/*.fpprofile %t/nokappa/ && cp %t/kappa/*.fpprofile %t/diverged/ && cp %t/kappa/*.fpprofile %t/noise/
; RUN: sed -i -e "s/^Kappa = .*/Kappa = 0/" -e "s/^KappaMark = ok/KappaMark = below-noise/" %t/noise/matmul_body.fpprofile %t/noise/fms_body.fpprofile
; RUN: sed -i "/^Kappa/d" %t/nokappa/matmul_body.fpprofile %t/nokappa/fms_body.fpprofile
; RUN: sed -i "s/^KappaMark = ok/KappaMark = diverged/" %t/diverged/matmul_body.fpprofile
; RUN: %{opt} -poseidon-profile-use=%t/kappa -poseidon-early-prune -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false -poseidon-cache=%t/c1 2>&1 | FileCheck --check-prefix=KAPPA %s
; RUN: %{opt} -poseidon-profile-use=%t/nokappa -poseidon-early-prune -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false -poseidon-cache=%t/c2 2>&1 | FileCheck --check-prefix=PLAIN %s
; RUN: %{opt} -poseidon-profile-use=%t/diverged -poseidon-early-prune -poseidon-enable-precision=false -poseidon-cache=%t/c3 2>&1 | FileCheck --check-prefix=DIVERGED %s
; RUN: %{opt} -poseidon-profile-use=%t/diverged -poseidon-early-prune -poseidon-enable-precision=false -poseidon-tau=1e-3 -poseidon-cache=%t/c5 2>&1 | FileCheck --check-prefix=DIVERGED %s
; RUN: %{opt} -poseidon-profile-use=%t/diverged -poseidon-early-prune -poseidon-enable-precision=false -poseidon-solve-jointly -poseidon-cache=%t/c6 2>&1 | FileCheck --check-prefix=DIVERGED %s
; RUN: %{opt} -poseidon-profile-use=%t/noise -poseidon-early-prune -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false -poseidon-cache=%t/c4 > %t/noise.log 2>&1
; RUN: FileCheck --check-prefix=PLAIN %s < %t/noise.log
; RUN: FileCheck --check-prefix=NOISE %s < %t/noise.log

; KAPPA: [poseidon] preprocess_fms_body: measured condition number 1.000000e-03 (ok), profiled sensitivity 1.187943e+04, error scale 8.417909e-08
; KAPPA: Initial AccuracyCost: 2.164332e-20
; KAPPA-NEXT: Initial ComputationCost: 2.564000e+03
; KAPPA: accuracy delta
; KAPPA-NEXT: ---
; KAPPA-NEXT: {{^}}1.428768e-11 -1.816320e+06 All FP64(0%) + FP32(100%){{$}}
; KAPPA: {{^}}6.795448e-19 -3.333120e+05 All FP64(0%) + Expansion2(100%){{$}}
; KAPPA: [poseidon] preprocess_matmul_body: measured condition number 1.000000e+03 (ok), profiled sensitivity 4.903078e+04, error scale 2.039535e-02
; KAPPA-NEXT: [poseidon] matmul[0] candidates (initial cost=5.120000e+02, executions=8192, globalM=16, globalN=16, globalK=0, K=16, gridCTAs=1):
; KAPPA-NEXT: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=3.332236e+01
; KAPPA-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06 dacc=2.978463e-02
; KAPPA-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06 dacc=2.540700e-02

; Without the Kappa lines every accuracy cost is unscaled; computation costs
; are the same.
; PLAIN-NOT: measured condition number
; PLAIN: Initial AccuracyCost: 2.571104e-13
; PLAIN-NEXT: Initial ComputationCost: 2.564000e+03
; PLAIN: accuracy delta
; PLAIN-NEXT: ---
; PLAIN-NEXT: {{^}}1.697296e-04 -1.816320e+06 All FP64(0%) + FP32(100%){{$}}
; PLAIN: {{^}}8.072609e-12 -3.333120e+05 All FP64(0%) + Expansion2(100%){{$}}
; PLAIN: [poseidon] matmul[0] candidates (initial cost=5.120000e+02, executions=8192, globalM=16, globalN=16, globalK=0, K=16, gridCTAs=1):
; PLAIN-NEXT: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=1.633821e+03
; PLAIN-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06 dacc=1.460363e+00
; PLAIN-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06 dacc=1.245725e+00

; A condition number below the runs' noise is ignored, with a warning.
; NOISE: warning: {{.*}}the profiling runs did not resolve the condition number of preprocess_fms_body against the declared metric from their noise (Kappa = 0, below-noise), so the site is solved against its own target; measure it with larger perturbation steps in POSEIDON_KAPPA_LADDER{{$}}
; NOISE: warning: {{.*}}condition number of preprocess_matmul_body

; A site whose kappa run diverged is left as written under a budget, a target
; and a joint solve.
; DIVERGED: warning: {{.*}}Poseidon: the profiling runs of preprocess_matmul_body diverged under the smallest perturbation, so no candidate can be traded against its accuracy and the site is left as written{{$}}
; DIVERGED: [poseidon] preprocess_matmul_body: measured condition number 1.000000e+03 (diverged), profiled sensitivity 4.903078e+04, error scale 1.000000e+00
; DIVERGED-NOT: Applying solution for matmul
; DIVERGED: no rewrite applied for preprocess_matmul_body

source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11
}

define dso_local void @fms_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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
define dso_local ptx_kernel void @_Z10fms_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fms_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}
declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
