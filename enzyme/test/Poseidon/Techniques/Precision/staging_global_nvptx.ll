; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/staging_global -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/f32 -poseidon-pin=_Z5stagePKdS0_Pd:P0_0,_Z5splitPKdS0_PdPf:P0_0 \
; RUN:   -o %t/f32.ll 2> %t/f32.log
; RUN: FileCheck --check-prefix=F32-LOG %s < %t/f32.log
; RUN: FileCheck --check-prefix=F32 %s < %t/f32.ll
; RUN: %{opt} -poseidon-cache=%t/ds -poseidon-pin=_Z5stagePKdS0_Pd:P0_2,_Z5splitPKdS0_PdPf:P0_2 \
; RUN:   -o %t/ds.ll 2> %t/ds.log
; RUN: FileCheck --check-prefix=DS-LOG %s < %t/ds.log
; RUN: FileCheck --check-prefix=DS %s < %t/ds.ll
; A __shared__ buffer the site reads directly, staged once and read by a
; contraction. The FP32 tier halves its stride and the df64 tier stores and
; loads {hi, lo} pairs. In @split a staged value is also split by the source
; itself, and the FP32 rounding the split extends is read again: that buffer
; keeps its FP64 layout.
; The IR is written by hand in the shape clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 gives for
;   __global__ POSEIDON_OPTIMIZE void stage(const double *__restrict__ x,
;                                           const double *__restrict__ b,
;                                           double *__restrict__ y) {
;     __shared__ double sm[4];
;     const int tx = threadIdx.x;
;     sm[tx] = x[tx];
;     __syncthreads();
;     double u = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 4; ++k)
;       u += sm[k] * b[k * 4 + tx];
;     y[tx] += u;
;   }
; and for split, which is stage followed by
;     const double v = sm[tx];
;     const float hi = (float)v;
;     lo[tx] = (float)(v - (double)hi);
;     y[tx + 4] = (double)hi * 2.0;
; The profiles are written by hand for one launch <<<1, 4>>> on
; x[i] = 1.0 + 0.01 * i and b[i] = 0.5 + 0.03 * i, with y zero.

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

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
