; RUN: rm -rf %t && mkdir -p %t/nokappa %t/diverged && cp %S/Inputs/site_kappa/*.fpprofile %t/nokappa/ && cp %S/Inputs/site_kappa/*.fpprofile %t/diverged/
; RUN: sed -i "/^Kappa/d" %t/nokappa/preprocess_matmul_body.fpprofile %t/nokappa/preprocess_fms_body.fpprofile
; RUN: sed -i "s/^KappaMark = ok/KappaMark = diverged/" %t/diverged/preprocess_matmul_body.fpprofile
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/site_kappa -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/c1 -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t.kappa.out 2>&1
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/nokappa -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/c2 -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t.plain.out 2>&1
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/diverged -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/c3 -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t.div.out 2>&1
; RUN: FileCheck --check-prefix=KAPPA %s < %t.kappa.out
; RUN: FileCheck --check-prefix=PLAIN %s < %t.plain.out
; RUN: FileCheck --check-prefix=DIVERGED %s < %t.div.out
; REQUIRES: poseidon, scev-trunc-iv-nowrap
; Two annotated sites in one module, each carrying its own condition number in
; its profile header: the accuracy cost of EVERY candidate class of a site is
; multiplied by that site's Kappa divided by the total sensitivity its profile
; records, the matrix product's candidates and the scalar subgraph's alike, and
; a site whose probe run diverged is scaled by a large finite factor instead so
; no candidate of it is ever traded against precision. The two bodies and their
; profiles are the Inputs/raise_tcec and Inputs/ds_expansion fixtures (real
; profiled runs on an RTX 5090); the Kappa values, 1000 on the product and
; 0.001 on the scalar subgraph, are chosen so the scaling is legible in the
; printed costs, and the second RUN is the same solve against the same profiles
; with the two header lines deleted.

; ModuleID = 'tcec_matmul.cu'
source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8, !tbaa !9
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8, !tbaa !9
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8, !tbaa !9
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11, !llvm.loop !11
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @fms_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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
; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z10fms_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !5
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !5
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fms_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}
; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !5
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !5
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
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
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.unroll.disable"}

; KAPPA: [poseidon] site preprocess_fms_body: kappa=1.000000e-03 (ok) sumsens=1.187943e+04 factor=8.417909e-08
; KAPPA: Initial AccuracyCost: 2.164332e-20
; KAPPA: Δ AccCost
; KAPPA-NEXT: ---
; KAPPA-NEXT: 1.428768e-11{{.*}}All FP64(0%) + FP32(100%)
; KAPPA: 5.708979e-19{{.*}}All FP64(0%) + Expansion2(100%)
; KAPPA: [poseidon] site preprocess_matmul_body: kappa=1.000000e+03 (ok) sumsens=4.903078e+04 factor=2.039535e-02
; KAPPA: Matmul[0] candidates (initial cost=4.988291e+04
; KAPPA-NEXT: wmma m16n16k16 f16/f32: Δcost=-380908769 ΔaccCost=3.332236e+01
; KAPPA-NEXT: tcec n=2 wmma m16n16k16 f16/f32: Δcost=-374564621 ΔaccCost=2.978463e-02
; KAPPA-NEXT: tcec n=3 wmma m16n16k16 f16/f32: Δcost=-368505296 ΔaccCost=2.540700e-02

; The same solve with the two header lines gone: every accuracy cost is the
; unscaled one, and the computation costs are untouched in both runs.
; PLAIN-NOT: kappa=
; PLAIN: Initial AccuracyCost: 2.571104e-13
; PLAIN: Δ AccCost
; PLAIN-NEXT: ---
; PLAIN-NEXT: 1.697296e-04{{.*}}All FP64(0%) + FP32(100%)
; PLAIN: 6.781944e-12{{.*}}All FP64(0%) + Expansion2(100%)
; PLAIN: Matmul[0] candidates (initial cost=4.988291e+04
; PLAIN-NEXT: wmma m16n16k16 f16/f32: Δcost=-380908769 ΔaccCost=1.633821e+03
; PLAIN-NEXT: tcec n=2 wmma m16n16k16 f16/f32: Δcost=-374564621 ΔaccCost=1.460363e+00
; PLAIN-NEXT: tcec n=3 wmma m16n16k16 f16/f32: Δcost=-368505296 ΔaccCost=1.245725e+00

; A site whose probe run diverged carries no measured kappa; its candidates are
; priced at 1e30, the order of the exponent penalty, so a joint budget is spent
; on any other site before this one's precision is traded away.
; DIVERGED: [poseidon] site preprocess_matmul_body: kappa=1.000000e+03 (diverged) sumsens=4.903078e+04 factor=1.000000e+30
; DIVERGED: Matmul[0] candidates (initial cost=4.988291e+04
; DIVERGED-NEXT: wmma m16n16k16 f16/f32: Δcost=-380908769 ΔaccCost=1.633821e+33
; DIVERGED-NEXT: tcec n=2 wmma m16n16k16 f16/f32: Δcost=-374564621 ΔaccCost=1.460363e+30
; DIVERGED-NEXT: tcec n=3 wmma m16n16k16 f16/f32: Δcost=-368505296 ΔaccCost=1.245725e+30
