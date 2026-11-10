; RUN: rm -rf %t.off && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/staging_param -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t.off -poseidon-apply-rewrites=PT0_1 -poseidon-print -S 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: rm -rf %t.ds && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/staging_param -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-cache=%t.ds -poseidon-apply-rewrites=PT0_2 -poseidon-print -S 2>&1 | FileCheck --check-prefix=DS %s
; RUN: rm -rf %t.ir && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/staging_param -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-cache=%t.ir -poseidon-apply-rewrites=PT0_2 -S 2>/dev/null | FileCheck --check-prefix=IR %s
; RUN: rm -rf %t.f32 && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/staging_param -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-cache=%t.f32 -poseidon-apply-rewrites=PT0_0 -poseidon-print -S 2>&1 | FileCheck --check-prefix=F32 %s
; REQUIRES: poseidon
; Two contractions staged through one __shared__ buffer that reaches the body
; as a pointer parameter, then a once-per-thread epilogue. Off by default the
; first contraction is a trivial 1-op subgraph and the buffer keeps its FP64
; layout; with the four arms the contractions join one optimization unit priced
; per loop depth, and the chosen tier narrows the buffer through the parameter.
; IR and profile from Inputs/staging_param/staged_body.cu (clang -O2
; -ffp-contract=on --cuda-gpu-arch=sm_120, profiled on an RTX 5090).

; ModuleID = 'Inputs/staging_param/staged_body.cu'
source_filename = "Inputs/staging_param/staged_body.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@_ZZ13staged_kernelPKdS0_PdS1_E1s = internal addrspace(3) global [256 x double] undef, align 8
@_ZZ13staged_kernelPKdS0_PdS1_E2ds = internal addrspace(3) global [256 x double] undef, align 8
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @staged_body(ptr noalias nofree noundef captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef readonly captures(none) %2, ptr noalias nofree noundef captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %7 = shl nuw nsw i32 %6, 4
  br label %12

8:                                                ; preds = %12
  %9 = add nuw nsw i32 %7, %5
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %10
  store double %24, ptr %11, align 8, !tbaa !10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %31

12:                                               ; preds = %4, %12
  %13 = phi i32 [ 0, %4 ], [ %25, %12 ]
  %14 = phi double [ 0.000000e+00, %4 ], [ %24, %12 ]
  %15 = add nuw nsw i32 %13, %7
  %16 = zext nneg i32 %15 to i64
  %17 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %16
  %18 = load double, ptr %17, align 8, !tbaa !10
  %19 = shl nuw nsw i32 %13, 4
  %20 = add nuw nsw i32 %19, %5
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %21
  %23 = load double, ptr %22, align 8, !tbaa !10
  %24 = tail call double @llvm.fmuladd.f64(double %18, double %23, double %14)
  %25 = add nuw nsw i32 %13, 1
  %26 = icmp eq i32 %25, 16
  br i1 %26, label %8, label %12, !llvm.loop !12

27:                                               ; preds = %31
  %28 = getelementptr inbounds nuw [8 x i8], ptr %3, i64 %10
  %29 = load double, ptr %28, align 8, !tbaa !10
  %30 = fadd double %43, %29
  store double %30, ptr %28, align 8, !tbaa !10
  ret void

31:                                               ; preds = %8, %31
  %32 = phi i32 [ 0, %8 ], [ %44, %31 ]
  %33 = phi double [ 0.000000e+00, %8 ], [ %43, %31 ]
  %34 = or disjoint i32 %32, %7
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %35
  %37 = load double, ptr %36, align 8, !tbaa !10
  %38 = shl nuw nsw i32 %32, 4
  %39 = add nuw nsw i32 %38, %5
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !10
  %43 = tail call double @llvm.fmuladd.f64(double %37, double %42, double %33)
  %44 = add nuw nsw i32 %32, 1
  %45 = icmp eq i32 %44, 16
  br i1 %45, label %27, label %31, !llvm.loop !15
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13staged_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add nuw nsw i32 %6, %7
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E2ds to ptr), i64 %9
  store double 0.000000e+00, ptr %10, align 8, !tbaa !10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !16
  %12 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !16
  tail call void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef nonnull @staged_body, i32 noundef %11, ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr), ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E2ds to ptr), i32 noundef %12, ptr noundef %0, i32 noundef %12, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

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
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unroll.disable"}
!15 = distinct !{!15, !13, !14}
!16 = !{!7, !7, i64 0}

; OFF: Skipping trivial subgraph
; OFF: [poseidon] Found 1 initial subgraphs in preprocess_staged_body
; OFF-NOT: merged {{[0-9]+}} FP subgraphs
; OFF-NOT: frequency-heterogeneous subgraph
; OFF-NOT: narrowed param-staged shared buffer
; OFF: Applying solution for CS:

; DS: Keeping 1-op reduction subgraph: {{.*}}@llvm.fmuladd.f64
; DS: [poseidon] Found 2 initial subgraphs in preprocess_staged_body
; DS: [poseidon] merged 2 FP subgraphs sharing 4 staging buffer(s) into one optimization unit (3 operations, 7 inputs, 3 outputs) in preprocess_staged_body
; DS: [poseidon] frequency-heterogeneous subgraph (normalizer 512, measured depth:exec 0:512 1:8192)
; DS: [poseidon] Selecting PT0_2: All FP64(0%) + Expansion2(100%)
; DS: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; DS: [poseidon] narrowed param-staged shared buffer @_ZZ13staged_kernelPKdS0_PdS1_E1s to df64 (1 loads, 1 stores)

; IR-LABEL: define {{.*}} @preprocess_staged_body(
; IR: store float %{{.+}}, ptr %[[HI:[0-9]+]], align 4
; IR-NEXT: %[[LO:[0-9]+]] = getelementptr i8, ptr %[[HI]], i64 4
; IR-NEXT: store float %{{.+}}, ptr %[[LO]], align 4
; IR-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; IR: load float, ptr %[[P:[^,]+]], align 4
; IR-NEXT: %[[Q:[0-9]+]] = getelementptr i8, ptr %[[P]], i64 4
; IR-NEXT: load float, ptr %[[Q]], align 4

; F32: [poseidon] Selecting PT0_0: All FP64(0%) + FP32(100%)
; F32: Applying solution for CS: All FP64(0%) + FP32(100%)
; F32: [poseidon] narrowed param-staged shared buffer @_ZZ13staged_kernelPKdS0_PdS1_E1s to fp32 (1 loads, 1 stores)
