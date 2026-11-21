;   Source:
;   extern "C" __device__ __attribute__((noinline)) void
;   matmul_body(const double *A, const double *B, double *D, const double *U,
;               const double *W, double *Y, double alpha) {
;     unsigned c = threadIdx.y * 16 + threadIdx.x;
;     double acc = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 16; ++k)
;       acc += A[threadIdx.y * 16 + k] * B[k * 16 + threadIdx.x];
;     D[c] = acc * alpha;
;     double u = U[c], w = W[c];
;     Y[c] = (sqrt(u + 1.0) - sqrt(u)) +
;            (((sinh(w) + asinh(w)) + tan(w)) + asin(w));
;   }
;   marked with __poseidon_fp_optimize from a 16x16 kernel.

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %D, ptr noundef readonly captures(none) %U, ptr noundef readonly captures(none) %W, ptr noundef writeonly captures(none) %Y, double noundef %alpha) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  %cell = add nuw nsw i32 %row, %tx
  %celli = zext nneg i32 %cell to i64
  br label %loop

exit:
  %d = fmul double %alpha, %acc.next
  %dp = getelementptr inbounds nuw [8 x i8], ptr %D, i64 %celli
  store double %d, ptr %dp, align 8
  %up = getelementptr inbounds nuw [8 x i8], ptr %U, i64 %celli
  %u = load double, ptr %up, align 8
  %wp = getelementptr inbounds nuw [8 x i8], ptr %W, i64 %celli
  %w = load double, ptr %wp, align 8
  %u1 = fadd double %u, 1.000000e+00
  %su1 = tail call double @llvm.sqrt.f64(double %u1)
  %su = tail call double @llvm.sqrt.f64(double %u)
  %p = fsub double %su1, %su
  %sh = tail call double @__nv_sinh(double %w)
  %ash = tail call double @__nv_asinh(double %w)
  %q1 = fadd double %sh, %ash
  %tn = tail call double @__nv_tan(double %w)
  %q2 = fadd double %q1, %tn
  %as = tail call double @__nv_asin(double %w)
  %q = fadd double %q2, %as
  %y = fadd double %p, %q
  %yp = getelementptr inbounds nuw [8 x i8], ptr %Y, i64 %celli
  store double %y, ptr %yp, align 8
  ret void

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %ai = add nuw nsw i32 %k, %row
  %aii = zext nneg i32 %ai to i64
  %ap = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aii
  %a = load double, ptr %ap, align 8
  %bk = shl nuw nsw i32 %k, 4
  %bi = add nuw nsw i32 %bk, %tx
  %bii = zext nneg i32 %bi to i64
  %bp = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bii
  %b = load double, ptr %bp, align 8
  %acc.next = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop, !llvm.loop !0
}

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_S0_S0_S1_S1_d(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, ptr noundef %6, ptr noundef %7, double noundef %8) #1 {
  %10 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iS1_iS1_iS2_S2_idEET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %10, ptr noundef %0, i32 noundef %10, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3, i32 noundef %10, ptr noundef %4, i32 noundef %10, ptr noundef %5, i32 noundef %11, ptr noundef %6, ptr noundef %7, i32 noundef %10, double noundef %8) #2
  ret void
}

declare void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iS1_iS1_iS2_S2_idEET_PvDpT0_(ptr, i32, ptr, i32, ptr, i32, ptr, ptr, i32, ptr, i32, ptr, i32, ptr, ptr, i32, double)
declare double @llvm.fmuladd.f64(double, double, double)
declare double @llvm.sqrt.f64(double)
declare double @__nv_sinh(double)
declare double @__nv_asinh(double)
declare double @__nv_tan(double)
declare double @__nv_asin(double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}
