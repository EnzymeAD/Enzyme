; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-print -S -poseidon-cache=%t/tau12 -poseidon-tau=1e-12 -o %t/tau12.ll 2> %t/tau12.err
; RUN: FileCheck --check-prefix=TAU12 %s < %t/tau12.err
; RUN: FileCheck --check-prefix=DESC12 %s < %t/tau12/_Z13matmul_kernelPKdS0_PdS1_.ozdispatch
; RUN: %clang -O0 -x ir %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/tau12 -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST12 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-print -S -poseidon-cache=%t/tau7 -poseidon-tau=1e-7 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU7 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-print -S -poseidon-cache=%t/tau10 -poseidon-tau=1e-10 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU10 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-print -S -poseidon-cache=%t/tau15 -poseidon-tau=1e-15 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU15 %s
; RUN: %clang -O0 -x ir %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/tau15 -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST15 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-print -S -poseidon-cache=%t/off -poseidon-tau=1e-12 -poseidon-ozaki-host-dispatch=false -o /dev/null 2>&1 | FileCheck --check-prefix=OFF %s
; REQUIRES: poseidon, scev-trunc-iv-nowrap
; Ozaki-II host dispatch selected by a tolerance: the scalar FP64 product of raise_scalar_tcec_nvptx.ll
; (same IR, same profile Inputs/raise_tcec) priced against the in-kernel raises and the host-dispatch
; families of Inputs/cm_gpu_fixture_dispatch_x1e6.csv. errorBudgetSelector takes the cheapest candidate
; whose modelled domain error clears tau, so the Ozaki-II moduli count rises with the tolerance; the
; device compile writes the .ozdispatch descriptor (last field 0 = Ozaki-II scheme, seventh field = nm)
; and the host compile of the same kernel's launch stub (Inputs/ozaki_dispatch/ozaki_stub_host.ll, from
; ozaki_stub.cu via clang -x cuda --cuda-host-only -O0 -S -emit-llvm) is rewritten to
; __poseidon_ozaki_dgemm with that nm. nm = 0 is the native cuBLAS DGEMM arm of the same entry.

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

; TAU12: [poseidon] matmul[0] pricing: baseline/MAC=4.988291e+04 executions=8192
; TAU12-NEXT: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.385254e+03  rel=6.786400e-02  domainError=3.137966e-04
; TAU12-NEXT: #1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.159686e+03  rel=8.338900e-02  domainError=1.572576e-07
; TAU12-NEXT: #2 tcec n=3 wmma m16n16k16 f16/f32  compCost/MAC=4.899349e+03  rel=9.821700e-02  domainError=5.874975e-08
; TAU12-NEXT: #3 ozaki-ii nm=8 wmma m16n16k16 s8/s32  compCost/MAC=1.550211e+03  rel=3.107700e-02  domainError=1.919601e-08
; TAU12-NEXT: #4 ozaki-ii nm=9 wmma m16n16k16 s8/s32  compCost/MAC=1.703202e+03  rel=3.414400e-02  domainError=1.237043e-09
; TAU12-NEXT: #5 ozaki-ii nm=10 wmma m16n16k16 s8/s32  compCost/MAC=1.865770e+03  rel=3.740300e-02  domainError=8.318238e-11
; TAU12-NEXT: #6 ozaki-ii nm=11 wmma m16n16k16 s8/s32  compCost/MAC=2.033826e+03  rel=4.077200e-02  domainError=5.322777e-12
; TAU12-NEXT: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.216647e+03  rel=4.443700e-02  domainError=3.148603e-13
; TAU12-NEXT: #8 ozaki-ii nm=13 wmma m16n16k16 s8/s32  compCost/MAC=2.386548e+03  rel=4.784300e-02  domainError=2.044637e-14
; TAU12-NEXT: #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=2.548917e+03  rel=5.109800e-02  domainError=5.085896e-15
; TAU12-NEXT: #10 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.518885e+03  rel=3.044900e-02  domainError=1.572576e-07
; TAU12-NEXT: #11 direct-dispatch cublas f16/f32  compCost/MAC=4.473499e+02  rel=8.968000e-03  domainError=3.137966e-04
; TAU12-NEXT: #12 direct-dispatch cublas bf16/f32  compCost/MAC=4.481979e+02  rel=8.985000e-03  domainError=2.340575e-03
; TAU12-NEXT: #13 direct-dispatch cublas tf32/f32  compCost/MAC=7.000068e+02  rel=1.403300e-02  domainError=3.137466e-04
; TAU12-NEXT: #14 native-dgemm cublas f64/f64  compCost/MAC=2.860994e+04  rel=5.735420e-01  domainError=0.000000e+00
; TAU12: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 3.148603e-13 <= 1.000000e-12 at confidence 9.500000e-01)
; TAU12: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7)
; TAU12: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; DESC12: _Z13matmul_kernelPKdS0_PdS1_ 2 0 1 16 1 12 16 16 16 16 16 16 0 0 0 0

; HOST12: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1 16x16x16)
; HOST12-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST12-NEXT: entry:
; HOST12-NEXT: call void @__poseidon_ozaki_dgemm(ptr %2, ptr %0, ptr %1, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr null, i32 12)
; HOST12-NEXT: ret void
; HOST12-NOT: cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; TAU7: Poseidon error-budget: matmul -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (domain err 1.919601e-08 <= 1.000000e-07 at confidence 9.500000e-01)
; TAU7: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; TAU7: wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=8 (standalone)

; TAU10: Poseidon error-budget: matmul -> ozaki-ii nm=10 wmma m16n16k16 s8/s32 (domain err 8.318238e-11 <= 1.000000e-10 at confidence 9.500000e-01)
; TAU10: Applying solution for matmul[0] -> ozaki-ii nm=10 wmma m16n16k16 s8/s32 (#5)
; TAU10: wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=10 (standalone)

; TAU15: Poseidon error-budget: matmul -> native-dgemm cublas f64/f64 (domain err 0.000000e+00 <= 1.000000e-15 at confidence 9.500000e-01)
; TAU15: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#14)
; TAU15: wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; HOST15: call void @__poseidon_ozaki_dgemm(ptr %2, ptr %0, ptr %1, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr null, i32 0)

; OFF-NOT: ozaki-ii
; OFF: Poseidon error-budget: matmul -> baseline F64 (no rewrite clears 1.000000e-12 at confidence 9.500000e-01 faster than F64)
; OFF-NOT: Applying solution for matmul
; OFF-NOT: wrote descriptor
