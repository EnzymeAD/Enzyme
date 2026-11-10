; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-comp-cost-budget=-1000000000000 -poseidon-print -S -o /dev/null -poseidon-cache=%t/prune 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-comp-cost-budget=-1000000000000 -poseidon-print -S -o /dev/null -poseidon-cache=%t/exact -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt
; REQUIRES: poseidon, scev-trunc-iv-nowrap

; A precision change priced against the rewrites taken. The kernel computes
; the matrix product with its epilogue D = acc * alpha, then Y, an FP subgraph
; the raise m does not touch. Herbie's cheaper answer x for Y
; (Inputs/exact_dp/reprice_herbie) is cheaper and more accurate than m alone,
; so pruning after the CM unit drops every strategy that raises m without
; rewriting Y. Y has one output, so its precision changes u change nothing
; once Y is rewritten, and u never meets m: the four cheapest points of the
; exact frontier, {u, m}, are missing. IR and profile are written by hand, the
; math functions as the libdevice calls the pass recognizes. A budget below
; every frontier point selects nothing, so only the solve runs.
;
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

; PRUNE: DP table contains 17 entries.
; PRUNE-BUDGETS-NOT: -1340878898

; EXACT: Exact DP: 21 frontier points
; EXACT: DP table contains 21 entries.
; EXACT-BUDGETS: {{^}}-1340878898,-1334534750,-1328475425,-1030015198,-999536151,
