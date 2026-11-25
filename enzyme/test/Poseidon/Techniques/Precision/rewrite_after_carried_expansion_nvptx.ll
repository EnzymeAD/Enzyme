; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_carried_expansion %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/rewrite_after_carried_expansion -poseidon-early-prune -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv "-poseidon-pin=P0=All FP64(0%%) + Expansion2(100%%),R2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2))" -poseidon-cache=%t/expansion-first -S -poseidon-print -o %t/expansion-first.ll 2>&1 | FileCheck --check-prefix=APPLY %s
; RUN: FileCheck --check-prefix=IR %s < %t/expansion-first.ll
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_carried_expansion %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/rewrite_after_carried_expansion -poseidon-early-prune -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv "-poseidon-pin=R2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2)),P0=All FP64(0%%) + Expansion2(100%%)" -poseidon-cache=%t/rewrite-first -S -poseidon-print -o %t/rewrite-first.ll 2>&1 | FileCheck --check-prefix=APPLY %s
; RUN: diff %t/expansion-first.ll %t/rewrite-first.ll

; APPLY: Applying solution for (* (- v1 v2) (+ v2 v1)) --(0)-> (-.f64 (*.f64 v1 v1) (*.f64 v2 v2))
; APPLY-NEXT: Applying solution for precision change All FP64(0%) + Expansion2(100%) (#3)

; IR-LABEL: define dso_local void @preprocess_carried_body(
; IR-SAME: ptr {{.*}}%0, double noundef %[[V2:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; IR-NOT: phi double
; IR: %[[V2SQ:[^ ]+]] = fmul double %[[V2]], %[[V2]]
; IR-NOT: phi double
; IR: %pair.phi.hi = phi float [ %[[NHI:[^ ]+]], %{{[0-9]+}} ], [ 0.000000e+00, %{{[0-9]+}} ]
; IR-NEXT: %pair.phi.lo = phi float [ %[[NLO:[^ ]+]], %{{[0-9]+}} ], [ 0.000000e+00, %{{[0-9]+}} ]
; IR-NEXT: %[[HI:.+]] = fpext float %pair.phi.hi to double
; IR-NEXT: %[[LO:.+]] = fpext float %pair.phi.lo to double
; IR-NEXT: %[[ACC:.+]] = fadd double %[[HI]], %[[LO]]
; IR-NOT: phi double
; IR: %herbie.mul = fmul double %[[ACC]], %[[ACC]]
; IR-NEXT: %herbie.sub = fsub double %herbie.mul, %[[V2SQ]]
; IR-NEXT: %[[IDX:[0-9]+]] = add {{.*}}i64
; IR-NEXT: %[[P:[0-9]+]] = getelementptr {{.*}}ptr %[[OUT]], i64 %[[IDX]]
; IR-NEXT: store double %herbie.sub, ptr %[[P]], align 8
; IR: %[[PH:[^ ]+]] = fmul float %pair.phi.hi, 5.000000e-01
; IR: call float @llvm.fma.f32(float %pair.phi.lo, float 5.000000e-01, float
; IR: %[[NHI]] = fadd float %[[TS:[^ ]+]], %[[T1:[^ ]+]]
; IR-NEXT: %[[D:[0-9]+]] = fsub float %[[TS]], %[[NHI]]
; IR-NEXT: %[[NLO]] = fadd float %[[T1]], %[[D]]
; IR-NOT: phi double
; IR: {{^}}}{{$}}

; A two-component expansion of one FP subgraph and a rewrite of another that
; reads the FP64 accumulator the expansion carries as a pair. In the loop of
; carried_body the accumulator %13 feeds the subgraph {%22 %23} that updates it (P0) and the
; subgraph {%14 %15 %16} that stores (acc - y) * (y + acc) (R2, Herbie's answer
; canned next to the profile). P0 carries %13 as two float PHIs, its users
; reading the pair joined back to binary64; the rewrite reads that same value as
; v1. The steps apply in the solve's order, the rewrite first, whatever the
; order of the items.
; The module: clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -S -emit-llvm carried.cu, with carried.cu
;   template <typename R, typename... T>
;   __device__ R __poseidon_fp_optimize(void *, T...);
;   __device__ int enzyme_const;
;   __device__ int enzyme_dup;
;   constexpr int K = 16;
;   extern "C" __device__ __attribute__((noinline)) void
;   carried_body(const double *x, double y, double *out) {
;     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
;     double acc = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < K; ++k) {
;       out[i * K + k] = (acc - y) * (acc + y);
;       double xi = x[i * K + k];
;       acc = acc * 0.5 + xi * xi;
;     }
;   }
;   __global__ void carried_kernel(const double *x, double y, double *out,
;                                  double *dout) {
;     __poseidon_fp_optimize<void>((void *)carried_body, enzyme_const, x,
;                                  enzyme_const, y, enzyme_dup, out, dout);
;   }
; Profile: the same source through poseidon-clang++ -poseidon-profile-generate,
; launched as carried_kernel<<<2, 64>>>(x, 0.75, out, dout) on
; x[i] = 0.25 + (i % 61) / 64.0, with dout set to 1.0.

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @carried_body(ptr nofree noundef readonly captures(none) %0, double noundef %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = shl i32 %8, 4
  br label %11

10:                                               ; preds = %11
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = fsub double %13, %1
  %15 = fadd double %1, %13
  %16 = fmul double %14, %15
  %17 = add nuw nsw i32 %12, %9
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %18
  store double %16, ptr %19, align 8
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %18
  %21 = load double, ptr %20, align 8
  %22 = fmul double %21, %21
  %23 = tail call double @llvm.fmuladd.f64(double %13, double 5.000000e-01, double %22)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %10, label %11
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z14carried_kernelPKddPdS1_(ptr noundef %0, double noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef nonnull @carried_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, double noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
