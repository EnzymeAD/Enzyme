; IR from clang -O2 -ffp-contract=on --cuda-gpu-arch=sm_120 --cuda-device-only:
;   extern "C" __device__ __attribute__((noinline)) void
;   matmul_body(const double *A, const double *B, double *D, double alpha) {
;     double acc = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 16; ++k)
;       acc += A[threadIdx.y * 16 + k] * B[k * 16 + threadIdx.x];
;     D[threadIdx.y * 16 + threadIdx.x] = acc * ((alpha + 1.0) - alpha);
;   }
; marked with __poseidon_fp_optimize from a 16x16 kernel; the profile is
; written by hand (alpha in [1e16, 1e17]), and Herbie's answer is replayed from
; Inputs/no_early_prune/refusal_herbie.

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2, double noundef %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %7 = shl nuw nsw i32 %6, 4
  br label %15

8:                                                ; preds = %15
  %9 = fadd double %3, 1.000000e+00
  %10 = fsub double %9, %3
  %11 = fmul double %10, %27
  %12 = add nuw nsw i32 %7, %5
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %13
  store double %11, ptr %14, align 8
  ret void

15:                                               ; preds = %4, %15
  %16 = phi i32 [ 0, %4 ], [ %28, %15 ]
  %17 = phi double [ 0.000000e+00, %4 ], [ %27, %15 ]
  %18 = add nuw nsw i32 %16, %7
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %19
  %21 = load double, ptr %20, align 8
  %22 = shl nuw nsw i32 %16, 4
  %23 = add nuw nsw i32 %22, %5
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %24
  %26 = load double, ptr %25, align 8
  %27 = tail call double @llvm.fmuladd.f64(double %21, double %26, double %17)
  %28 = add nuw nsw i32 %16, 1
  %29 = icmp eq i32 %28, 16
  br i1 %29, label %8, label %15
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_d(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4) local_unnamed_addr #2 {
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %7 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_idEET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %6, ptr noundef %0, i32 noundef %6, ptr noundef %1, i32 noundef %7, ptr noundef %2, ptr noundef %3, i32 noundef %6, double noundef %4) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_idEET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, double noundef) local_unnamed_addr #3

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
