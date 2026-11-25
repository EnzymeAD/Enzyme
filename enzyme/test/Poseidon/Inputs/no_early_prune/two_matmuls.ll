; IR from clang -O2 -ffp-contract=on --cuda-gpu-arch=sm_120 --cuda-device-only:
;   extern "C" __device__ __attribute__((noinline)) void
;   matmul_body(const double *A, const double *B, const double *C, double *D,
;               double *E) {
;     double d = 0.0, e = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 16; ++k) {
;       double a = A[threadIdx.y * 16 + k];
;       d += a * B[k * 16 + threadIdx.x];
;       e += a * C[k * 16 + threadIdx.x];
;     }
;     D[threadIdx.y * 16 + threadIdx.x] = d;
;     E[threadIdx.y * 16 + threadIdx.x] = e;
;   }
; marked with __poseidon_fp_optimize from a 16x16 kernel; the profile is
; written by hand (d's output gradient a tenth of e's).

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef readonly captures(none) %2, ptr nofree noundef writeonly captures(none) %3, ptr nofree noundef writeonly captures(none) %4) #0 {
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %8 = shl nuw nsw i32 %7, 4
  br label %14

9:                                                ; preds = %14
  %10 = add nuw nsw i32 %8, %6
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw [8 x i8], ptr %3, i64 %11
  store double %27, ptr %12, align 8
  %13 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %11
  store double %30, ptr %13, align 8
  ret void

14:                                               ; preds = %5, %14
  %15 = phi i32 [ 0, %5 ], [ %31, %14 ]
  %16 = phi double [ 0.000000e+00, %5 ], [ %30, %14 ]
  %17 = phi double [ 0.000000e+00, %5 ], [ %27, %14 ]
  %18 = add nuw nsw i32 %15, %8
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %19
  %21 = load double, ptr %20, align 8
  %22 = shl nuw nsw i32 %15, 4
  %23 = add nuw nsw i32 %22, %6
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %24
  %26 = load double, ptr %25, align 8
  %27 = tail call double @llvm.fmuladd.f64(double %21, double %26, double %17)
  %28 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %24
  %29 = load double, ptr %28, align 8
  %30 = tail call double @llvm.fmuladd.f64(double %21, double %29, double %16)
  %31 = add nuw nsw i32 %15, 1
  %32 = icmp eq i32 %31, 16
  br i1 %32, label %9, label %14
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_S0_PdS1_S1_S1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, ptr noundef %6) local_unnamed_addr #2 {
  %8 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %9 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iS1_iPdS2_iS2_S2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %8, ptr noundef %0, i32 noundef %8, ptr noundef %1, i32 noundef %8, ptr noundef %2, i32 noundef %9, ptr noundef %3, ptr noundef %4, i32 noundef %9, ptr noundef %5, ptr noundef %6) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iS1_iPdS2_iS2_S2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
