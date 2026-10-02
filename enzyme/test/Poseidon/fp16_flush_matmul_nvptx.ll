; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-cache=%t/normal 2>&1 | FileCheck --check-prefix=NORMAL %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/Inputs/fp16_range/matmul_tiny -poseidon-cache=%t/tiny 2>&1 | FileCheck --check-prefix=TINY %s
; REQUIRES: poseidon, scev-trunc-iv-nowrap
; FP16 flush pricing of a matrix-product raise. The product of raise_scalar_tcec_nvptx.ll (same IR) under
; two profiles: Inputs/raise_tcec (A in [1, 1.255]) and Inputs/fp16_range/matmul_tiny (A scaled by 1e-9,
; below FP16's smallest subnormal 5.96e-8). Rounding A to FP16 flushes it to zero, so the direct FP16 raise
; models a p95 relative error of 1 and the tolerance 1e-3 that takes it on the first profile refuses it on
; the second; every candidate whose operands pass through the FP16 exponent range (the direct raise and
; both TCEC chains) also has -poseidon-exponent-penalty (1e30) added to its accuracy cost.
; matmul_tiny profile: Inputs/raise_tcec/tcec_matmul.cu with hA[i] = 1e-9 * (1.0 + 0.001 * i), built with
; the RUN lines of Integration/raise_scalar_matmul_f64.cu (Poseidon + Enzyme plugins,
; -poseidon-profile-generate, FPProfilerCUDA.cu, libposeidon_profile.a) and run on an RTX 5090.

; REQUIRES: poseidon, scev-trunc-iv-nowrap
; Scalar FMA reduction loop raised in-kernel: direct and TCEC classes priced from the wmma_inkernel_rel fixture rows, the most accurate cost-reducing one (TCEC n=3) materialized; IR and profile from Inputs/raise_tcec/tcec_matmul.cu (clang -O2 -ffp-contract=on --cuda-gpu-arch=sm_120, profiled on an RTX 5090).

; ModuleID = 'tcec_matmul.cu'
source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8, !tbaa !9
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8, !tbaa !9
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8, !tbaa !9
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11, !llvm.loop !11
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !5
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !5
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git a70419505471bd8240ef3451dcdd541f8676477c)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !7, i64 0}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.unroll.disable"}

; NORMAL: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; NORMAL: wmma m16n16k16 f16/f32: Δcost=-380908769 ΔaccCost=9.867296e+00
; NORMAL-NEXT: tcec n=2 wmma m16n16k16 f16/f32: Δcost=-374564621 ΔaccCost=5.093297e-03
; NORMAL-NEXT: tcec n=3 wmma m16n16k16 f16/f32: Δcost=-368505296 ΔaccCost=1.896120e-03
; NORMAL: Poseidon error-budget: matmul -> wmma m16n16k16 f16/f32 (domain err 3.137966e-04 <= 1.000000e-03 at confidence 9.500000e-01)
; NORMAL: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)

; TINY: DOMERR matmul in=f16 acc=f32 p95_relErr=1.000000e+00
; TINY: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.385254e+03  rel=6.786400e-02  domainError=INF
; TINY: #1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.159686e+03  rel=8.338900e-02  domainError=INF
; TINY: wmma m16n16k16 f16/f32: Δcost=-380908769 ΔaccCost=1.000000e+30
; TINY-NEXT: tcec n=2 wmma m16n16k16 f16/f32: Δcost=-374564621 ΔaccCost=1.000000e+30
; TINY-NEXT: tcec n=3 wmma m16n16k16 f16/f32: Δcost=-368505296 ΔaccCost=1.000000e+30
; TINY-NOT: error-budget: matmul -> wmma m16n16k16 f16/f32
; TINY-NOT: Applying solution for matmul[0]
; TINY: error-budget: matmul -> baseline F64 (no rewrite clears 1.000000e-03
; TINY: Finished optimizing preprocess_matmul_body
