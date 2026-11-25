; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/staging_param -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/f32 -poseidon-pin=P0_0 -o %t/f32.ll 2> %t/f32.log
; RUN: FileCheck --check-prefix=LOG %s < %t/f32.log
; RUN: FileCheck --check-prefix=IR %s < %t/f32.ll
; RUN: %{opt} -poseidon-cache=%t/ds -poseidon-pin=P0_2 -o %t/ds.ll 2> %t/ds.log
; RUN: FileCheck --check-prefix=LOG %s < %t/ds.log
; RUN: FileCheck --check-prefix=IR %s < %t/ds.ll

; The kernel of staging_param_nvptx.ll, which also hands its __shared__ buffer
; s to a function that reads it as doubles in the same launch: s keeps its FP64
; layout under every tier.

; LOG: Applying solution for precision change
; LOG-NOT: narrowed param-staged shared buffer

; IR-LABEL: define {{.*}} @_Z13staged_kernelPKdS0_PdS1_(
; IR: call void @ext_reader(ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr))
; IR-LABEL: define {{.*}} @preprocess_staged_body(
; IR: store double %{{.+}}, ptr %{{.+}}, align 8
; IR-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; IR: load double, ptr %{{.+}}, align 8

; ModuleID = 'Techniques/Precision/Inputs/staging_param/staged_body.cu'
source_filename = "Techniques/Precision/Inputs/staging_param/staged_body.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@_ZZ13staged_kernelPKdS0_PdS1_E1s = internal addrspace(3) global [256 x double] undef, align 8
@_ZZ13staged_kernelPKdS0_PdS1_E2ds = internal addrspace(3) global [256 x double] undef, align 8
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @staged_body(ptr noalias nofree noundef captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef readonly captures(none) %2, ptr noalias nofree noundef captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %7 = shl nuw nsw i32 %6, 4
  br label %12

8:                                                ; preds = %12
  %9 = add nuw nsw i32 %7, %5
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %10
  store double %24, ptr %11, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %31

12:                                               ; preds = %4, %12
  %13 = phi i32 [ 0, %4 ], [ %25, %12 ]
  %14 = phi double [ 0.000000e+00, %4 ], [ %24, %12 ]
  %15 = add nuw nsw i32 %13, %7
  %16 = zext nneg i32 %15 to i64
  %17 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %16
  %18 = load double, ptr %17, align 8
  %19 = shl nuw nsw i32 %13, 4
  %20 = add nuw nsw i32 %19, %5
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %21
  %23 = load double, ptr %22, align 8
  %24 = tail call double @llvm.fmuladd.f64(double %18, double %23, double %14)
  %25 = add nuw nsw i32 %13, 1
  %26 = icmp eq i32 %25, 16
  br i1 %26, label %8, label %12

27:                                               ; preds = %31
  %28 = getelementptr inbounds nuw [8 x i8], ptr %3, i64 %10
  %29 = load double, ptr %28, align 8
  %30 = fadd double %43, %29
  store double %30, ptr %28, align 8
  ret void

31:                                               ; preds = %8, %31
  %32 = phi i32 [ 0, %8 ], [ %44, %31 ]
  %33 = phi double [ 0.000000e+00, %8 ], [ %43, %31 ]
  %34 = or disjoint i32 %32, %7
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %35
  %37 = load double, ptr %36, align 8
  %38 = shl nuw nsw i32 %32, 4
  %39 = add nuw nsw i32 %38, %5
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %40
  %42 = load double, ptr %41, align 8
  %43 = tail call double @llvm.fmuladd.f64(double %37, double %42, double %33)
  %44 = add nuw nsw i32 %32, 1
  %45 = icmp eq i32 %44, 16
  br i1 %45, label %27, label %31
}

declare double @llvm.fmuladd.f64(double, double, double) #1

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define dso_local ptx_kernel void @_Z13staged_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add nuw nsw i32 %6, %7
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E2ds to ptr), i64 %9
  store double 0.000000e+00, ptr %10, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  %12 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  call void @ext_reader(ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr))
  tail call void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef nonnull @staged_body, i32 noundef %11, ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr), ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E2ds to ptr), i32 noundef %12, ptr noundef %0, i32 noundef %12, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

declare void @ext_reader(ptr)

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
