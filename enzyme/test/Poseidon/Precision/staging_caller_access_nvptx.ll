; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/staging_caller_access \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/f32 -poseidon-apply-rewrites=direct_body:PT0_0,views_body:PT0_0 \
; RUN:   -o %t/f32.ll 2> %t/f32.log
; RUN: FileCheck --check-prefix=LOG %s < %t/f32.log
; RUN: FileCheck --check-prefix=IR %s < %t/f32.ll
; RUN: %{opt} -poseidon-cache=%t/ds -poseidon-apply-rewrites=direct_body:PT0_2,views_body:PT0_2 \
; RUN:   -o %t/ds.ll 2> %t/ds.log
; RUN: FileCheck --check-prefix=LOG %s < %t/ds.log
; RUN: FileCheck --check-prefix=IR %s < %t/ds.ll

; Two per-call sites whose bodies read a __shared__ buffer directly, one
; through views of one stride and one through views of two, while the kernel
; that calls each writes that buffer itself before the call. The kernel's
; accesses run in the same launch at the FP64 layout, so neither buffer is
; narrowed under any tier.
; The IR is written by hand in the shape clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 gives for
;   __shared__ double sm_direct[4], sm_views[8];
;   __device__ void direct_body(const double *b, double *y) {
;     const int tx = threadIdx.x;
;     double u = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 4; ++k)
;       u += sm_direct[k] * b[k * 4 + tx];
;     y[tx] += u;
;   }
;   __global__ void direct(const double *x, const double *b, double *y,
;                          double *dy) {
;     sm_direct[threadIdx.x] = x[threadIdx.x];
;     __syncthreads();
;     __poseidon_fp_optimize<void>((void *)direct_body, enzyme_const, b,
;                                  enzyme_dup, y, dy);
;   }
; and for views_body, which also reads ((double(*)[2])sm_views)[tx][0] after
; the loop and multiplies the sum by it, with the kernel views writing both
; doubles of row tx of that view. The profiles are written by hand for one
; launch <<<1, 4>>> on x[i] = 1.0 + 0.01 * i and b[i] = 0.5 + 0.03 * i, with
; y zero.

; LOG: [poseidon] Selecting direct_body:PT0_{{[02]}}
; LOG: Applying solution for CS:
; LOG: [poseidon] Selecting views_body:PT0_{{[02]}}
; LOG: Applying solution for CS:
; LOG-NOT: narrowed

; IR-LABEL: define dso_local ptx_kernel void @_Z6directPKdS0_PdS1_(
; IR: store double %{{.+}}, ptr %{{.+}}, align 8
; IR-LABEL: define dso_local ptx_kernel void @_Z5viewsPKdS0_PdS1_(
; IR: store double %{{.+}}, ptr %{{.+}}, align 8
; IR-LABEL: define {{.*}} @preprocess_direct_body(
; IR: load double, ptr addrspacecast (ptr addrspace(3) @sm_direct to ptr), align 8
; IR-LABEL: define {{.*}} @preprocess_views_body(
; IR: load double, ptr addrspacecast (ptr addrspace(3) @sm_views to ptr), align 8

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

!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
