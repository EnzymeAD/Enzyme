; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/two_matmuls -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/prune 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-TABLE %s < %t/prune/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/two_matmuls -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/exact -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-TABLE %s < %t/exact/table.json

; Two matrix products on one loop. The loop carries two reduction phis, so it
; yields two MatmulUnits with the same footprint, at most one of them
; raised. At K = 16 the native DGEMM dispatch has zero modelled error and
; dominates the empty strategy for the first product (the e reduction);
; pruning after it leaves no strategy without e, and the second (the d
; reduction, ten times less sensitive) is refused everywhere. The exact solve
; finds d's options, each ten times more accurate at the same cost. A budget
; below every frontier point selects nothing, so only the solve runs.
;
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

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @matmul_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef readonly captures(none) %2, ptr nofree noundef writeonly captures(none) %3, ptr nofree noundef writeonly captures(none) %4) #0 {
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %8 = shl nuw nsw i32 %7, 4
  br label %14

9:                                                ; preds = %14
  %10 = add nuw nsw i32 %8, %6
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw [8 x i8], ptr %3, i64 %11
  store double %27, ptr %12, align 8, !tbaa !10
  %13 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %11
  store double %30, ptr %13, align 8, !tbaa !10
  ret void

14:                                               ; preds = %5, %14
  %15 = phi i32 [ 0, %5 ], [ %31, %14 ]
  %16 = phi double [ 0.000000e+00, %5 ], [ %30, %14 ]
  %17 = phi double [ 0.000000e+00, %5 ], [ %27, %14 ]
  %18 = add nuw nsw i32 %15, %8
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %19
  %21 = load double, ptr %20, align 8, !tbaa !10
  %22 = shl nuw nsw i32 %15, 4
  %23 = add nuw nsw i32 %22, %6
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %24
  %26 = load double, ptr %25, align 8, !tbaa !10
  %27 = tail call double @llvm.fmuladd.f64(double %21, double %26, double %17)
  %28 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %24
  %29 = load double, ptr %28, align 8, !tbaa !10
  %30 = tail call double @llvm.fmuladd.f64(double %21, double %29, double %16)
  %31 = add nuw nsw i32 %15, 1
  %32 = icmp eq i32 %31, 16
  br i1 %32, label %9, label %14, !llvm.loop !12
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_S0_PdS1_S1_S1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, ptr noundef %6) local_unnamed_addr #2 {
  %8 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !15
  %9 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !15
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iS1_iPdS2_iS2_S2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %8, ptr noundef %0, i32 noundef %8, ptr noundef %1, i32 noundef %8, ptr noundef %2, i32 noundef %9, ptr noundef %3, ptr noundef %4, i32 noundef %9, ptr noundef %5, ptr noundef %6) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iS1_iPdS2_iS2_S2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unroll.disable"}
!15 = !{!7, !7, i64 0}

; PRUNE: DP table contains 10 entries.
; PRUNE-TABLE: "-6479617209": 1633.8211240338183{{[0-9]*}}
; PRUNE-TABLE-NOT: "itemIndex": 1

; EXACT: Exact DP: 10 frontier points
; EXACT: DP table contains 10 entries.
; EXACT-TABLE: "-6479617209": 163.38211240338188{{[0-9]*}}
; EXACT-TABLE: "itemIndex": 1
