; RUN: rm -rf %t.off && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/staging_multiview -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-cache=%t.off -poseidon-apply-rewrites=PT0_0 -poseidon-print -S 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: rm -rf %t.f32 && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/staging_multiview -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-narrow-multiview-staging -poseidon-cache=%t.f32 -poseidon-apply-rewrites=PT0_0 -poseidon-print -S 2>&1 | FileCheck --check-prefix=F32 %s
; RUN: rm -rf %t.ds && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/staging_multiview -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-narrow-multiview-staging -poseidon-cache=%t.ds -poseidon-apply-rewrites=PT0_2 -poseidon-print -S 2>&1 | FileCheck --check-prefix=DS %s
; RUN: rm -rf %t.ir && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/staging_multiview -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-reduction-subgraphs -poseidon-merge-shared-staging -poseidon-freq-weighted-pricing -poseidon-narrow-param-staging -poseidon-narrow-multiview-staging -poseidon-cache=%t.ir -poseidon-apply-rewrites=PT0_2 -S 2>/dev/null | FileCheck --check-prefix=IR %s
; REQUIRES: poseidon
; One __shared__ array used through views of two shapes, one at a constant
; offset, by contractions that read each staged value twice (MFEM's
; SmemPADiffusionApply3D). Element GEPs of three strides make the
; global-rooted narrowing give up, and without the multiview arm the buffer
; keeps its FP64 layout; with it the chosen tier narrows every view, and the
; df64 tier rewires both splits of a value read twice to one pair load.
; IR and profile from Inputs/staging_multiview/multiview_kernel.cu (clang -O2
; -ffp-contract=on --cuda-gpu-arch=sm_120, profiled on an RTX 5090).

; OFF: [narrow] G=@_ZZ9multiviewPKdS0_PdE2sm casts=1 elemGEPs=3 strideK=8 bail=1
; OFF: Applying solution for CS: All FP64(0%) + FP32(100%)
; OFF-NOT: narrowed multiview shared buffer

; F32: Applying solution for CS: All FP64(0%) + FP32(100%)
; F32: [poseidon] narrowed multiview shared buffer @_ZZ9multiviewPKdS0_PdE2sm to fp32 (12 loads, 3 stores)

; DS: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; DS: [poseidon] narrowed multiview shared buffer @_ZZ9multiviewPKdS0_PdE2sm to df64 (12 loads, 3 stores)

; IR-LABEL: define internal void @preprocess__Z9multiviewPKdS0_Pd_poseidon_body(
; IR: store float
; IR-NEXT: [[LO:%[0-9]+]] = getelementptr i8, ptr [[SLOT:%[0-9]+]], i64 4
; IR-NEXT: store float {{%[0-9]+}}, ptr [[LO]], align 4
; IR: load float
; IR-NEXT: getelementptr i8, ptr {{%[0-9]+}}, i64 4
; IR-NEXT: load float

; ModuleID = 'Inputs/staging_multiview/multiview_kernel.cu'
source_filename = "Inputs/staging_multiview/multiview_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ9multiviewPKdS0_PdE2sm = internal unnamed_addr addrspace(3) global [2 x [64 x double]] undef, align 8
@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [45 x i8] c"Inputs/staging_multiview/multiview_kernel.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z9multiviewPKdS0_Pd, ptr @.str, ptr @.str1, i32 10, ptr null }], section "llvm.metadata"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z9multiviewPKdS0_Pd(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = shl nuw nsw i32 %6, 4
  %8 = shl nuw nsw i32 %5, 2
  %9 = add nuw nsw i32 %8, %4
  %10 = add nuw nsw i32 %9, %7
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %11
  %13 = load double, ptr %12, align 8, !tbaa !10
  %14 = zext nneg i32 %5 to i64
  %15 = getelementptr inbounds nuw [32 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 512), i64 %14
  %16 = zext nneg i32 %4 to i64
  %17 = getelementptr inbounds nuw [8 x i8], ptr %15, i64 %16
  store double %13, ptr %17, align 8, !tbaa !10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %18 = load double, ptr %15, align 8, !tbaa !10
  %19 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %16
  %20 = load double, ptr %19, align 8, !tbaa !10
  %21 = tail call double @llvm.fmuladd.f64(double %18, double %20, double 0.000000e+00)
  %22 = zext nneg i32 %4 to i64
  %23 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 128
  %25 = load double, ptr %24, align 8, !tbaa !10
  %26 = tail call double @llvm.fmuladd.f64(double %18, double %25, double 0.000000e+00)
  %27 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %28 = load double, ptr %27, align 8, !tbaa !10
  %29 = add nuw nsw i32 %4, 4
  %30 = zext nneg i32 %29 to i64
  %31 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %30
  %32 = load double, ptr %31, align 8, !tbaa !10
  %33 = tail call double @llvm.fmuladd.f64(double %28, double %32, double %21)
  %34 = zext nneg i32 %4 to i64
  %35 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %34
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 160
  %37 = load double, ptr %36, align 8, !tbaa !10
  %38 = tail call double @llvm.fmuladd.f64(double %28, double %37, double %26)
  %39 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %40 = load double, ptr %39, align 8, !tbaa !10
  %41 = zext nneg i32 %4 to i64
  %42 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 64
  %44 = load double, ptr %43, align 8, !tbaa !10
  %45 = tail call double @llvm.fmuladd.f64(double %40, double %44, double %33)
  %46 = zext nneg i32 %4 to i64
  %47 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %46
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 192
  %49 = load double, ptr %48, align 8, !tbaa !10
  %50 = tail call double @llvm.fmuladd.f64(double %40, double %49, double %38)
  %51 = getelementptr inbounds nuw i8, ptr %15, i64 24
  %52 = load double, ptr %51, align 8, !tbaa !10
  %53 = zext nneg i32 %4 to i64
  %54 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 96
  %56 = load double, ptr %55, align 8, !tbaa !10
  %57 = tail call double @llvm.fmuladd.f64(double %52, double %56, double %45)
  %58 = zext nneg i32 %4 to i64
  %59 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %58
  %60 = getelementptr inbounds nuw i8, ptr %59, i64 224
  %61 = load double, ptr %60, align 8, !tbaa !10
  %62 = tail call double @llvm.fmuladd.f64(double %52, double %61, double %50)
  %63 = getelementptr inbounds nuw [64 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %14
  %64 = getelementptr inbounds nuw [8 x i8], ptr %63, i64 %16
  store double %57, ptr %64, align 8, !tbaa !10
  %65 = getelementptr inbounds nuw [8 x i8], ptr %63, i64 %30
  store double %62, ptr %65, align 8, !tbaa !10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %66 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %16
  %67 = load double, ptr %66, align 8, !tbaa !10
  %68 = zext nneg i32 %8 to i64
  %69 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %68
  %70 = load double, ptr %69, align 8, !tbaa !10
  %71 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %30
  %72 = load double, ptr %71, align 8, !tbaa !10
  %73 = zext nneg i32 %8 to i64
  %74 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 128
  %76 = load double, ptr %75, align 8, !tbaa !10
  %77 = fmul double %72, %76
  %78 = tail call double @llvm.fmuladd.f64(double %67, double %70, double %77)
  %79 = fadd double %78, 0.000000e+00
  %80 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 64), i64 %16
  %81 = load double, ptr %80, align 8, !tbaa !10
  %82 = zext nneg i32 %8 to i64
  %83 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %82
  %84 = getelementptr inbounds nuw i8, ptr %83, i64 8
  %85 = load double, ptr %84, align 8, !tbaa !10
  %86 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 64), i64 %30
  %87 = load double, ptr %86, align 8, !tbaa !10
  %88 = zext nneg i32 %8 to i64
  %89 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %88
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 136
  %91 = load double, ptr %90, align 8, !tbaa !10
  %92 = fmul double %87, %91
  %93 = tail call double @llvm.fmuladd.f64(double %81, double %85, double %92)
  %94 = fadd double %79, %93
  %95 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 128), i64 %16
  %96 = load double, ptr %95, align 8, !tbaa !10
  %97 = zext nneg i32 %8 to i64
  %98 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %97
  %99 = getelementptr inbounds nuw i8, ptr %98, i64 16
  %100 = load double, ptr %99, align 8, !tbaa !10
  %101 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 128), i64 %30
  %102 = load double, ptr %101, align 8, !tbaa !10
  %103 = zext nneg i32 %8 to i64
  %104 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %103
  %105 = getelementptr inbounds nuw i8, ptr %104, i64 144
  %106 = load double, ptr %105, align 8, !tbaa !10
  %107 = fmul double %102, %106
  %108 = tail call double @llvm.fmuladd.f64(double %96, double %100, double %107)
  %109 = fadd double %94, %108
  %110 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 192), i64 %16
  %111 = load double, ptr %110, align 8, !tbaa !10
  %112 = zext nneg i32 %8 to i64
  %113 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %112
  %114 = getelementptr inbounds nuw i8, ptr %113, i64 24
  %115 = load double, ptr %114, align 8, !tbaa !10
  %116 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 192), i64 %30
  %117 = load double, ptr %116, align 8, !tbaa !10
  %118 = zext nneg i32 %8 to i64
  %119 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %118
  %120 = getelementptr inbounds nuw i8, ptr %119, i64 152
  %121 = load double, ptr %120, align 8, !tbaa !10
  %122 = fmul double %117, %121
  %123 = tail call double @llvm.fmuladd.f64(double %111, double %115, double %122)
  %124 = fadd double %109, %123
  %125 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %11
  %126 = load double, ptr %125, align 8, !tbaa !10
  %127 = fadd double %124, %126
  store double %127, ptr %125, align 8, !tbaa !10
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
