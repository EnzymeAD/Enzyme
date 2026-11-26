; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; DEFINE: %{global} = %t/global.ll -poseidon-profile-use=%S/Inputs/staging_global
; DEFINE: %{caller} = %t/caller.ll -poseidon-profile-use=%S/Inputs/staging_caller_access
; RUN: rm -rf %t && split-file %s %t
; RUN: %{opt} %{global} -poseidon-cache=%t/f32 -poseidon-pin=_Z5stagePKdS0_Pd:P0_0,_Z5splitPKdS0_PdPf:P0_0 -o %t/f32.ll 2> %t/f32.log
; RUN: FileCheck --check-prefix=F32-LOG %s < %t/f32.log
; RUN: FileCheck --check-prefix=F32 %s < %t/f32.ll
; RUN: %{opt} %{global} -poseidon-cache=%t/ds -poseidon-pin=_Z5stagePKdS0_Pd:P0_2,_Z5splitPKdS0_PdPf:P0_2 -o %t/ds.ll 2> %t/ds.log
; RUN: FileCheck --check-prefix=DS-LOG %s < %t/ds.log
; RUN: FileCheck --check-prefix=DS %s < %t/ds.ll
; RUN: %{opt} %{caller} -poseidon-cache=%t/caller-f32 -poseidon-pin=direct_body:P0_0,views_body:P0_0 -o %t/caller-f32.ll 2> %t/caller-f32.log
; RUN: FileCheck --check-prefix=CALLER-LOG %s < %t/caller-f32.log
; RUN: FileCheck --check-prefix=CALLER-IR %s < %t/caller-f32.ll
; RUN: %{opt} %{caller} -poseidon-cache=%t/caller-ds -poseidon-pin=direct_body:P0_2,views_body:P0_2 -o %t/caller-ds.ll 2> %t/caller-ds.log
; RUN: FileCheck --check-prefix=CALLER-LOG %s < %t/caller-ds.log
; RUN: FileCheck --check-prefix=CALLER-IR %s < %t/caller-ds.ll

; A __shared__ buffer a site reads directly. global.ll: staged once and read by
; a contraction, the FP32 tier halves its stride and the df64 tier stores and
; loads {hi, lo} pairs; in @split the source itself splits a staged value and
; reads the FP32 rounding again, so that buffer keeps its FP64 layout.
; caller.ll: two per-call sites whose calling kernel writes the buffer before
; the call, at the FP64 layout in the same launch, so no tier narrows it.
; Hand-written IR in clang's -O2 sm_120 shape; hand-written profiles of one
; launch <<<1, 4>>>.

; F32-LOG: [poseidon] Selecting _Z5stagePKdS0_Pd:P0_0: All FP64(0%) + FP32(100%)
; F32-LOG: [poseidon] narrowed shared staging buffer @_ZZ5stagePKdS0_PdE2sm (1 loads, 1 stores, stride 8->4)
; F32-LOG: [poseidon] applied shared-staging narrowing in preprocess__Z5stagePKdS0_Pd_poseidon_body
; F32-LOG: [poseidon] Selecting _Z5splitPKdS0_PdPf:P0_0: All FP64(0%) + FP32(100%)
; F32-LOG-NOT: narrowed {{.*}}@_ZZ5splitPKdS0_PdPfE2sm
; F32-LABEL: define internal void @preprocess__Z5stagePKdS0_Pd_poseidon_body(
; F32: [[SLOT:%[0-9]+]] = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), i64
; F32-NEXT: [[XF:%[0-9]+]] = fptrunc double {{%[0-9]+}} to float
; F32-NEXT: store float [[XF]], ptr [[SLOT]], align 4
; F32: load float, ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), align 4
; F32-LABEL: define internal void @preprocess__Z5splitPKdS0_PdPf_poseidon_body(
; F32: store double {{%[0-9]+}}, ptr [[SSLOT:%[0-9]+]], align 8
; F32: %v = load double, ptr [[SSLOT]], align 8

; DS-LOG: [poseidon] Selecting _Z5stagePKdS0_Pd:P0_2: All FP64(0%) + Expansion2(100%)
; DS-LOG: [poseidon] narrowed multiview shared buffer @_ZZ5stagePKdS0_PdE2sm to df64 (1 loads, 1 stores)
; DS-LOG: [poseidon] Selecting _Z5splitPKdS0_PdPf:P0_2: All FP64(0%) + Expansion2(100%)
; DS-LOG-NOT: narrowed {{.*}}@_ZZ5splitPKdS0_PdPfE2sm
; DS-LABEL: define internal void @preprocess__Z5stagePKdS0_Pd_poseidon_body(
; DS: [[SLOT:%[0-9]+]] = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), i64
; DS: store float {{%[0-9]+}}, ptr [[SLOT]], align 4
; DS-NEXT: [[LOP:%[0-9]+]] = getelementptr i8, ptr [[SLOT]], i64 4
; DS-NEXT: store float {{%[0-9]+}}, ptr [[LOP]], align 4
; DS: load float, ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), align 4
; DS-NEXT: load float, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), i64 4), align 4
; DS-LABEL: define internal void @preprocess__Z5splitPKdS0_PdPf_poseidon_body(
; DS: store double {{%[0-9]+}}, ptr [[SSLOT:%[0-9]+]], align 8
; DS: %v = load double, ptr [[SSLOT]], align 8
; DS-NEXT: %hi = fptrunc double %v to float
; DS-NEXT: %hib = fpext float %hi to double
; DS-NEXT: %los = fsub double %v, %hib
; DS-NEXT: %lo = fptrunc double %los to float
; DS-NEXT: %w = fmul double %hib, 2.000000e+00

; CALLER-LOG: [poseidon] Selecting direct_body:P0_{{[02]}}
; CALLER-LOG: Applying solution for precision change
; CALLER-LOG: [poseidon] Selecting views_body:P0_{{[02]}}
; CALLER-LOG: Applying solution for precision change
; CALLER-LOG-NOT: narrowed

; CALLER-IR-LABEL: define dso_local ptx_kernel void @_Z6directPKdS0_PdS1_(
; CALLER-IR: store double %{{.+}}, ptr %{{.+}}, align 8
; CALLER-IR-LABEL: define dso_local ptx_kernel void @_Z5viewsPKdS0_PdS1_(
; CALLER-IR: store double %{{.+}}, ptr %{{.+}}, align 8
; CALLER-IR-LABEL: define {{.*}} @preprocess_direct_body(
; CALLER-IR: load double, ptr addrspacecast (ptr addrspace(3) @sm_direct to ptr), align 8
; CALLER-IR-LABEL: define {{.*}} @preprocess_views_body(
; CALLER-IR: load double, ptr addrspacecast (ptr addrspace(3) @sm_views to ptr), align 8

;--- global.ll
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@_ZZ5stagePKdS0_PdE2sm = internal unnamed_addr addrspace(3) global [4 x double] undef, align 8
@_ZZ5splitPKdS0_PdPfE2sm = internal unnamed_addr addrspace(3) global [4 x double] undef, align 8
@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [61 x i8] c"Techniques/Precision/Inputs/staging_global/staged_kernels.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z5stagePKdS0_Pd, ptr @.str, ptr @.str1, i32 9, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @_Z5splitPKdS0_PdPf, ptr @.str, ptr @.str1, i32 24, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z5stagePKdS0_Pd(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = zext nneg i32 %4 to i64
  %6 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %5
  %7 = load double, ptr %6, align 8
  %8 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), i64 %5
  store double %7, ptr %8, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %9

9:
  %10 = phi i32 [ 0, %3 ], [ %20, %9 ]
  %11 = phi double [ 0.000000e+00, %3 ], [ %19, %9 ]
  %12 = zext nneg i32 %10 to i64
  %13 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5stagePKdS0_PdE2sm to ptr), i64 %12
  %14 = load double, ptr %13, align 8
  %15 = shl nuw nsw i32 %10, 2
  %16 = add nuw nsw i32 %15, %4
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %17
  %b = load double, ptr %18, align 8
  %19 = tail call double @llvm.fmuladd.f64(double %14, double %b, double %11)
  %20 = add nuw nsw i32 %10, 1
  %21 = icmp eq i32 %20, 4
  br i1 %21, label %22, label %9

22:
  %23 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %5
  %24 = load double, ptr %23, align 8
  %25 = fadd double %19, %24
  store double %25, ptr %23, align 8
  ret void
}

define dso_local ptx_kernel void @_Z5splitPKdS0_PdPf(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef captures(none) %2, ptr noalias nofree noundef captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %6
  %8 = load double, ptr %7, align 8
  %9 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5splitPKdS0_PdPfE2sm to ptr), i64 %6
  store double %8, ptr %9, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %10

10:
  %11 = phi i32 [ 0, %4 ], [ %21, %10 ]
  %12 = phi double [ 0.000000e+00, %4 ], [ %20, %10 ]
  %13 = zext nneg i32 %11 to i64
  %14 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ5splitPKdS0_PdPfE2sm to ptr), i64 %13
  %15 = load double, ptr %14, align 8
  %16 = shl nuw nsw i32 %11, 2
  %17 = add nuw nsw i32 %16, %5
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %18
  %b = load double, ptr %19, align 8
  %20 = tail call double @llvm.fmuladd.f64(double %15, double %b, double %12)
  %21 = add nuw nsw i32 %11, 1
  %22 = icmp eq i32 %21, 4
  br i1 %22, label %23, label %10

23:
  %24 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %6
  %25 = load double, ptr %24, align 8
  %26 = fadd double %20, %25
  store double %26, ptr %24, align 8
  %v = load double, ptr %9, align 8
  %hi = fptrunc double %v to float
  %hib = fpext float %hi to double
  %los = fsub double %v, %hib
  %lo = fptrunc double %los to float
  %w = fmul double %hib, 2.000000e+00
  %27 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %6
  store float %lo, ptr %27, align 4
  %28 = getelementptr inbounds nuw i8, ptr %24, i64 32
  store double %w, ptr %28, align 8
  ret void
}

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1
declare double @llvm.fmuladd.f64(double, double, double) #2
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

;--- caller.ll
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@sm_direct = internal addrspace(3) global [4 x double] undef, align 8
@sm_views = internal addrspace(3) global [8 x double] undef, align 8
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @direct_body(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef captures(none) %1) #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = zext nneg i32 %3 to i64
  br label %5

5:
  %6 = phi i32 [ 0, %2 ], [ %15, %5 ]
  %7 = phi double [ 0.000000e+00, %2 ], [ %14, %5 ]
  %8 = zext nneg i32 %6 to i64
  %9 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @sm_direct to ptr), i64 %8
  %10 = load double, ptr %9, align 8
  %11 = shl nuw nsw i32 %6, 2
  %12 = add nuw nsw i32 %11, %3
  %13 = zext nneg i32 %12 to i64
  %bp = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %13
  %b = load double, ptr %bp, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %10, double %b, double %7)
  %15 = add nuw nsw i32 %6, 1
  %16 = icmp eq i32 %15, 4
  br i1 %16, label %17, label %5

17:
  %18 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %4
  %19 = load double, ptr %18, align 8
  %20 = fadd double %14, %19
  store double %20, ptr %18, align 8
  ret void
}

define dso_local ptx_kernel void @_Z6directPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %6
  %8 = load double, ptr %7, align 8
  %9 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @sm_direct to ptr), i64 %6
  store double %8, ptr %9, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %10 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiPdS2_EET_PvDpT0_(ptr noundef nonnull @direct_body, i32 noundef %10, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3) #5
  ret void
}

define dso_local void @views_body(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef captures(none) %1) #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = zext nneg i32 %3 to i64
  br label %5

5:
  %6 = phi i32 [ 0, %2 ], [ %15, %5 ]
  %7 = phi double [ 0.000000e+00, %2 ], [ %14, %5 ]
  %8 = zext nneg i32 %6 to i64
  %9 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @sm_views to ptr), i64 %8
  %10 = load double, ptr %9, align 8
  %11 = shl nuw nsw i32 %6, 2
  %12 = add nuw nsw i32 %11, %3
  %13 = zext nneg i32 %12 to i64
  %bp = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %13
  %b = load double, ptr %bp, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %10, double %b, double %7)
  %15 = add nuw nsw i32 %6, 1
  %16 = icmp eq i32 %15, 4
  br i1 %16, label %17, label %5

17:
  %wp = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @sm_views to ptr), i64 %4
  %w = load double, ptr %wp, align 8
  %t = fmul double %w, %14
  %18 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %4
  %19 = load double, ptr %18, align 8
  %20 = fadd double %t, %19
  store double %20, ptr %18, align 8
  ret void
}

define dso_local ptx_kernel void @_Z5viewsPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds nuw [16 x i8], ptr %0, i64 %6
  %8 = load double, ptr %7, align 8
  %9 = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @sm_views to ptr), i64 %6
  store double %8, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store double %11, ptr %12, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %13 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %14 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiPdS2_EET_PvDpT0_(ptr noundef nonnull @views_body, i32 noundef %13, ptr noundef %1, i32 noundef %14, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
