; DEFINE: %{opt} = %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/staging_param -poseidon-early-prune -poseidon-enable-algebraic=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %s -poseidon-cache=%t/off -poseidon-pin=P0_1 -poseidon-print 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: %{opt} %s -poseidon-subgraph-reductions -poseidon-subgraph-merge-staging -poseidon-cache=%t/ds -poseidon-pin=P0_2 -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=DS %s
; RUN: %{opt} %s -poseidon-subgraph-reductions -poseidon-subgraph-merge-staging -poseidon-precision-frequency-weighted -poseidon-cache=%t/ir -poseidon-pin=P0_2 2>&1 \
; RUN:   | FileCheck --check-prefix=IR %s
; RUN: %{opt} %s -poseidon-subgraph-reductions -poseidon-subgraph-merge-staging -poseidon-precision-frequency-weighted -poseidon-cache=%t/f32 -poseidon-pin=P0_0 -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=F32 %s
; RUN: sed -e '/^  tail call void @_Z22__poseidon_fp_optimize/i\  call void @ext_reader(ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr))' \
; RUN:   -e '$a declare void @ext_reader(ptr)' %s > %t/ext.ll
; RUN: %{opt} %t/ext.ll -poseidon-print -poseidon-cache=%t/ext-f32 -poseidon-pin=P0_0 -o %t/ext-f32.ll 2> %t/ext-f32.log
; RUN: FileCheck --check-prefix=EXT-LOG %s < %t/ext-f32.log
; RUN: FileCheck --check-prefix=EXT-IR %s < %t/ext-f32.ll
; RUN: %{opt} %t/ext.ll -poseidon-print -poseidon-cache=%t/ext-ds -poseidon-pin=P0_2 -o %t/ext-ds.ll 2> %t/ext-ds.log
; RUN: FileCheck --check-prefix=EXT-LOG %s < %t/ext-ds.log
; RUN: FileCheck --check-prefix=EXT-IR %s < %t/ext-ds.ll

; Two contractions staged through a __shared__ buffer that reaches the body as a
; pointer parameter: the chosen tier narrows the buffer through the parameter;
; with the subgraph flags the contractions join one unit priced per loop depth.
; A kernel that also hands the buffer to an external reader keeps it FP64.
; Module of staged_body.cu (staged_body, staged_kernel) via clang -x cuda
; --cuda-device-only --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on; profile from
; two launches <<<1, dim3(16, 16)>>> on an sm_120 GPU.

; OFF: Skipping trivial subgraph
; OFF: [poseidon] Found 1 initial subgraphs in preprocess_staged_body
; OFF-NOT: merged {{[0-9]+}} FP subgraphs
; OFF-NOT: frequency-heterogeneous subgraph
; OFF: Applying solution for precision change Funcs FP64(0%) + FP32(100%)
; OFF: [poseidon] narrowed param-staged shared buffer @_ZZ13staged_kernelPKdS0_PdS1_E1s to fp32 (1 loads, 1 stores)

; DS: Keeping 1-op reduction subgraph: {{.*}}@llvm.fmuladd.f64
; DS: [poseidon] Found 2 initial subgraphs in preprocess_staged_body
; DS: [poseidon] merged 2 FP subgraphs sharing 4 staging buffer(s) into one optimization unit (3 operations, 7 inputs, 3 outputs) in preprocess_staged_body
; DS: [poseidon] frequency-heterogeneous subgraph (normalizer 512, measured depth:exec 0:512 1:8192)
; DS: [poseidon] Selecting P0_2: All FP64(0%) + Expansion2(100%)
; DS: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; DS: [poseidon] narrowed param-staged shared buffer @_ZZ13staged_kernelPKdS0_PdS1_E1s to df64 (1 loads, 1 stores)

; IR-LABEL: define {{.*}} @preprocess_staged_body(
; IR: store float %{{.+}}, ptr %[[HI:[0-9]+]], align 4
; IR-NEXT: %[[LO:[0-9]+]] = getelementptr i8, ptr %[[HI]], i64 4
; IR-NEXT: store float %{{.+}}, ptr %[[LO]], align 4
; IR-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; IR: load float, ptr %[[P:[^,]+]], align 4
; IR-NEXT: %[[Q:[0-9]+]] = getelementptr i8, ptr %[[P]], i64 4
; IR-NEXT: load float, ptr %[[Q]], align 4

; F32: [poseidon] Selecting P0_0: All FP64(0%) + FP32(100%)
; F32: Applying solution for precision change All FP64(0%) + FP32(100%)
; F32: [poseidon] narrowed param-staged shared buffer @_ZZ13staged_kernelPKdS0_PdS1_E1s to fp32 (1 loads, 1 stores)


; EXT-LOG: Applying solution for precision change
; EXT-LOG-NOT: narrowed param-staged shared buffer

; EXT-IR-LABEL: define {{.*}} @_Z13staged_kernelPKdS0_PdS1_(
; EXT-IR: call void @ext_reader(ptr addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr))
; EXT-IR-LABEL: define {{.*}} @preprocess_staged_body(
; EXT-IR: store double %{{.+}}, ptr %{{.+}}, align 8
; EXT-IR-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; EXT-IR: load double, ptr %{{.+}}, align 8

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
  tail call void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef nonnull @staged_body, i32 noundef %11, ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E1s to ptr), ptr noundef addrspacecast (ptr addrspace(3) @_ZZ13staged_kernelPKdS0_PdS1_E2ds to ptr), i32 noundef %12, ptr noundef %0, i32 noundef %12, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPdS0_iPKdiS2_iS0_S0_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4
attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
