; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/staging_multiview \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-reduction-subgraphs -poseidon-merge-shared-staging \
; DEFINE:   -poseidon-freq-weighted-pricing -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/f32 -poseidon-apply-rewrites=PT0_0 -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=F32 %s
; RUN: %{opt} -poseidon-cache=%t/ds -poseidon-apply-rewrites=PT0_2 -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=DS %s
; RUN: %{opt} -poseidon-cache=%t/ir -poseidon-apply-rewrites=PT0_2 2>&1 | FileCheck --check-prefix=IR %s
; One __shared__ array used through views of two shapes, one at a constant
; offset, by contractions that read each staged value twice. Element GEPs of
; three strides are no single stride to halve, but the chosen tier narrows
; every view, and the df64 tier rewires both splits of a value read twice to
; one pair load.
; The IR is from
;   __global__ POSEIDON_OPTIMIZE void multiview(const double *__restrict__ x,
;                                               const double *__restrict__ b,
;                                               double *__restrict__ y) {
;     __shared__ double sm[2][64];
;     double(*X)[4] = (double(*)[4])(sm + 1);
;     double(*T)[8] = (double(*)[8])(sm + 0);
;     const int tx = threadIdx.x, ty = threadIdx.y, e = blockIdx.x;
;     X[ty][tx] = x[e * 16 + ty * 4 + tx];
;     __syncthreads();
;     double u = 0.0, v = 0.0;
;   #pragma unroll
;     for (int k = 0; k < 4; ++k) {
;       u += X[ty][k] * b[k * 4 + tx];
;       v += X[ty][k] * b[16 + k * 4 + tx];
;     }
;     T[ty][tx] = u;
;     T[ty][4 + tx] = v;
;     __syncthreads();
;     double w = 0.0;
;   #pragma unroll
;     for (int k = 0; k < 4; ++k)
;       w += T[k][tx] * b[ty * 4 + k] + T[k][4 + tx] * b[16 + ty * 4 + k];
;     y[e * 16 + ty * 4 + tx] += w;
;   }
; via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -S -emit-llvm, two contractions staged in shared memory.
; The profile: the same source through poseidon-clang++
; -poseidon-profile-generate, launched as multiview<<<8, dim3(4, 4)>>> on
; x[i] = 1.0 + 0.01 * i and b[i] = 0.5 + 0.03 * i.

; F32: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; F32: [poseidon] narrowed multiview shared buffer @_ZZ9multiviewPKdS0_PdE2sm to fp32 (12 loads, 3 stores)
; F32-NEXT: [poseidon] applied param-staged shared narrowing in preprocess__Z9multiviewPKdS0_Pd_poseidon_body
; F32-LABEL: ; ModuleID =
; F32-LABEL: define internal void @preprocess__Z9multiviewPKdS0_Pd_poseidon_body(
; F32: [[X:%[0-9]+]] = load double, ptr
; F32: [[ROW:%[0-9]+]] = getelementptr {{.*}}@_ZZ9multiviewPKdS0_PdE2sm{{.*}}, i64 256), i64
; F32-NEXT: [[TID:%[0-9]+]] = zext {{.*}} to i64
; F32-NEXT: [[SLOT:%[0-9]+]] = getelementptr {{.*}}ptr [[ROW]], i64 [[TID]]
; F32-NEXT: [[XF:%[0-9]+]] = fptrunc double [[X]] to float
; F32-NEXT: store float [[XF]], ptr [[SLOT]], align 4
; F32-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; F32-NEXT: [[V:%[0-9]+]] = load float, ptr [[ROW]], align 4
; F32: call float @llvm.fmuladd.f32(float [[V]], float
; F32: call float @llvm.fmuladd.f32(float [[V]], float
; F32: [[R1:%[0-9]+]] = getelementptr {{.*}}ptr getelementptr {{.*}}@_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 32), i64 [[TID]]
; F32-NEXT: load float, ptr [[R1]], align 4

; DS: Applying solution for CS: All FP64(0%) + Expansion2(100%) (#2)
; DS: [poseidon] narrowed multiview shared buffer @_ZZ9multiviewPKdS0_PdE2sm to df64 (12 loads, 3 stores)
; DS-NEXT: [poseidon] folded df64 join/split roundtrips at the staging boundary in preprocess__Z9multiviewPKdS0_Pd_poseidon_body

; IR-LABEL: define internal void @preprocess__Z9multiviewPKdS0_Pd_poseidon_body(
; IR: [[X:%[0-9]+]] = load double, ptr
; IR: [[ROW:%[0-9]+]] = getelementptr {{.*}}@_ZZ9multiviewPKdS0_PdE2sm{{.*}}, i64 512), i64
; IR-NEXT: [[TID:%[0-9]+]] = zext {{.*}} to i64
; IR-NEXT: [[SLOT:%[0-9]+]] = getelementptr {{.*}}ptr [[ROW]], i64 [[TID]]
; IR-NEXT: [[HI:%[0-9]+]] = fptrunc double [[X]] to float
; IR-NEXT: [[HIX:%[0-9]+]] = fpext float [[HI]] to double
; IR-NEXT: [[LOX:%[0-9]+]] = fsub double [[X]], [[HIX]]
; IR-NEXT: [[LO:%[0-9]+]] = fptrunc double [[LOX]] to float
; IR-NEXT: store float [[HI]], ptr [[SLOT]], align 4
; IR-NEXT: [[SLOTLO:%[0-9]+]] = getelementptr i8, ptr [[SLOT]], i64 4
; IR-NEXT: store float [[LO]], ptr [[SLOTLO]], align 4
; IR-NEXT: tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; IR-NEXT: [[VHI:%[0-9]+]] = load float, ptr [[ROW]], align 4
; IR-NEXT: [[VLOP:%[0-9]+]] = getelementptr i8, ptr [[ROW]], i64 4
; IR-NEXT: [[VLO:%[0-9]+]] = load float, ptr [[VLOP]], align 4
; IR-NOT: ptr [[ROW]], align
; IR: {{%[^ ]+}} = fmul float [[VHI]], [[B1:%[^ ]+]]
; IR: call float @llvm.fma.f32(float [[VHI]], float [[B1]], float
; IR-NEXT: call float @llvm.fma.f32(float [[VHI]], float {{%[^,]+}}, float
; IR-NEXT: call float @llvm.fma.f32(float [[VLO]], float [[B1]], float
; IR-NOT: ptr [[ROW]], align
; IR: {{%[^ ]+}} = fmul float [[VHI]], [[B2:%[^ ]+]]
; IR: call float @llvm.fma.f32(float [[VHI]], float [[B2]], float
; IR-NEXT: call float @llvm.fma.f32(float [[VHI]], float {{%[^,]+}}, float
; IR-NEXT: call float @llvm.fma.f32(float [[VLO]], float [[B2]], float

; ModuleID = 'Inputs/staging_multiview/multiview_kernel.cu'
source_filename = "Inputs/staging_multiview/multiview_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ9multiviewPKdS0_PdE2sm = internal unnamed_addr addrspace(3) global [2 x [64 x double]] undef, align 8
@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [45 x i8] c"Inputs/staging_multiview/multiview_kernel.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z9multiviewPKdS0_Pd, ptr @.str, ptr @.str1, i32 10, ptr null }], section "llvm.metadata"

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
  %13 = load double, ptr %12, align 8
  %14 = zext nneg i32 %5 to i64
  %15 = getelementptr inbounds nuw [32 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 512), i64 %14
  %16 = zext nneg i32 %4 to i64
  %17 = getelementptr inbounds nuw [8 x i8], ptr %15, i64 %16
  store double %13, ptr %17, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %18 = load double, ptr %15, align 8
  %19 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %16
  %20 = load double, ptr %19, align 8
  %21 = tail call double @llvm.fmuladd.f64(double %18, double %20, double 0.000000e+00)
  %22 = zext nneg i32 %4 to i64
  %23 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 128
  %25 = load double, ptr %24, align 8
  %26 = tail call double @llvm.fmuladd.f64(double %18, double %25, double 0.000000e+00)
  %27 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %28 = load double, ptr %27, align 8
  %29 = add nuw nsw i32 %4, 4
  %30 = zext nneg i32 %29 to i64
  %31 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %30
  %32 = load double, ptr %31, align 8
  %33 = tail call double @llvm.fmuladd.f64(double %28, double %32, double %21)
  %34 = zext nneg i32 %4 to i64
  %35 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %34
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 160
  %37 = load double, ptr %36, align 8
  %38 = tail call double @llvm.fmuladd.f64(double %28, double %37, double %26)
  %39 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %40 = load double, ptr %39, align 8
  %41 = zext nneg i32 %4 to i64
  %42 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 64
  %44 = load double, ptr %43, align 8
  %45 = tail call double @llvm.fmuladd.f64(double %40, double %44, double %33)
  %46 = zext nneg i32 %4 to i64
  %47 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %46
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 192
  %49 = load double, ptr %48, align 8
  %50 = tail call double @llvm.fmuladd.f64(double %40, double %49, double %38)
  %51 = getelementptr inbounds nuw i8, ptr %15, i64 24
  %52 = load double, ptr %51, align 8
  %53 = zext nneg i32 %4 to i64
  %54 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 96
  %56 = load double, ptr %55, align 8
  %57 = tail call double @llvm.fmuladd.f64(double %52, double %56, double %45)
  %58 = zext nneg i32 %4 to i64
  %59 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %58
  %60 = getelementptr inbounds nuw i8, ptr %59, i64 224
  %61 = load double, ptr %60, align 8
  %62 = tail call double @llvm.fmuladd.f64(double %52, double %61, double %50)
  %63 = getelementptr inbounds nuw [64 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %14
  %64 = getelementptr inbounds nuw [8 x i8], ptr %63, i64 %16
  store double %57, ptr %64, align 8
  %65 = getelementptr inbounds nuw [8 x i8], ptr %63, i64 %30
  store double %62, ptr %65, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %66 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %16
  %67 = load double, ptr %66, align 8
  %68 = zext nneg i32 %8 to i64
  %69 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %68
  %70 = load double, ptr %69, align 8
  %71 = getelementptr inbounds nuw [8 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 %30
  %72 = load double, ptr %71, align 8
  %73 = zext nneg i32 %8 to i64
  %74 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 128
  %76 = load double, ptr %75, align 8
  %77 = fmul double %72, %76
  %78 = tail call double @llvm.fmuladd.f64(double %67, double %70, double %77)
  %79 = fadd double %78, 0.000000e+00
  %80 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 64), i64 %16
  %81 = load double, ptr %80, align 8
  %82 = zext nneg i32 %8 to i64
  %83 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %82
  %84 = getelementptr inbounds nuw i8, ptr %83, i64 8
  %85 = load double, ptr %84, align 8
  %86 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 64), i64 %30
  %87 = load double, ptr %86, align 8
  %88 = zext nneg i32 %8 to i64
  %89 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %88
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 136
  %91 = load double, ptr %90, align 8
  %92 = fmul double %87, %91
  %93 = tail call double @llvm.fmuladd.f64(double %81, double %85, double %92)
  %94 = fadd double %79, %93
  %95 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 128), i64 %16
  %96 = load double, ptr %95, align 8
  %97 = zext nneg i32 %8 to i64
  %98 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %97
  %99 = getelementptr inbounds nuw i8, ptr %98, i64 16
  %100 = load double, ptr %99, align 8
  %101 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 128), i64 %30
  %102 = load double, ptr %101, align 8
  %103 = zext nneg i32 %8 to i64
  %104 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %103
  %105 = getelementptr inbounds nuw i8, ptr %104, i64 144
  %106 = load double, ptr %105, align 8
  %107 = fmul double %102, %106
  %108 = tail call double @llvm.fmuladd.f64(double %96, double %100, double %107)
  %109 = fadd double %94, %108
  %110 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 192), i64 %16
  %111 = load double, ptr %110, align 8
  %112 = zext nneg i32 %8 to i64
  %113 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %112
  %114 = getelementptr inbounds nuw i8, ptr %113, i64 24
  %115 = load double, ptr %114, align 8
  %116 = getelementptr inbounds nuw [8 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZZ9multiviewPKdS0_PdE2sm to ptr), i64 192), i64 %30
  %117 = load double, ptr %116, align 8
  %118 = zext nneg i32 %8 to i64
  %119 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %118
  %120 = getelementptr inbounds nuw i8, ptr %119, i64 152
  %121 = load double, ptr %120, align 8
  %122 = fmul double %117, %121
  %123 = tail call double @llvm.fmuladd.f64(double %111, double %115, double %122)
  %124 = fadd double %109, %123
  %125 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %11
  %126 = load double, ptr %125, align 8
  %127 = fadd double %124, %126
  store double %127, ptr %125, align 8
  ret void
}

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

declare double @llvm.fmuladd.f64(double, double, double) #2

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
