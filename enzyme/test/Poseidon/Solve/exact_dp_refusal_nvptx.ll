; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/refusal_herbie %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/exact_dp/refusal -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/prune 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/refusal_herbie %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/exact_dp/refusal -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/exact -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt

; Footprint refusal. The FP subgraph of the reduction loop's FMA also holds
; the epilogue D = acc * ((alpha + 1) - alpha), whose Herbie rewrite acc is
; cheaper and more accurate than the original. Pruning after the CO unit drops
; the empty strategy, the rewrite refuses the overlapping raise, and no raise
; reaches the frontier; the exact solve finds all three. A budget below every
; frontier point selects nothing, so only the solve runs.
;
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
; Inputs/exact_dp/refusal_herbie.

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
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
  store double %11, ptr %14, align 8, !tbaa !10
  ret void

15:                                               ; preds = %4, %15
  %16 = phi i32 [ 0, %4 ], [ %28, %15 ]
  %17 = phi double [ 0.000000e+00, %4 ], [ %27, %15 ]
  %18 = add nuw nsw i32 %16, %7
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %19
  %21 = load double, ptr %20, align 8, !tbaa !10
  %22 = shl nuw nsw i32 %16, 4
  %23 = add nuw nsw i32 %22, %5
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %24
  %26 = load double, ptr %25, align 8, !tbaa !10
  %27 = tail call double @llvm.fmuladd.f64(double %21, double %26, double %17)
  %28 = add nuw nsw i32 %16, 1
  %29 = icmp eq i32 %28, 16
  br i1 %29, label %8, label %15, !llvm.loop !12
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_d(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4) local_unnamed_addr #2 {
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !15
  %7 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !15
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_idEET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %6, ptr noundef %0, i32 noundef %6, ptr noundef %1, i32 noundef %7, ptr noundef %2, ptr noundef %3, i32 noundef %6, double noundef %4) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_idEET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, double noundef) local_unnamed_addr #3

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

; {{[^ -~]+}} stands for the Greek capital delta the log prints.
; PRUNE: Matmul[0] candidates
; PRUNE-NEXT: wmma m16n16k16 f16/f32: {{[^ -~]+}}cost=-3.932160e+06
; PRUNE-NEXT: tcec n=2 wmma m16n16k16 f16/f32: {{[^ -~]+}}cost=-3.866624e+06
; PRUNE-NEXT: tcec n=3 wmma m16n16k16 f16/f32: {{[^ -~]+}}cost=-3.801088e+06
; PRUNE: DP table contains 1 entries.
; PRUNE-BUDGETS: {{^}}-0.10526315789473684{{$}}

; EXACT: Exact DP: 4 frontier points
; EXACT: DP table contains 4 entries.
; EXACT-BUDGETS: {{^}}-0.78947368421052633,-0.77631578947368418,-0.76315789473684215,-0.10526315789473684{{$}}
