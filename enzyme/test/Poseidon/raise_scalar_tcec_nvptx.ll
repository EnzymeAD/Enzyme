; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t.out 2>&1
; RUN: FileCheck %s < %t.out
; RUN: FileCheck --check-prefix=CLEAN %s < %t.out
;
; The SAME matmul, selected by a tolerance written AT THE SITE (poseidon_tau on
; the marker) with NO -poseidon-tau and no cost budget: the site's own number
; drives errorBudgetSelector exactly as the flag does, so the picks and the
; modelled domain errors below are the ones the flag produces at the same value.
; sed makes the marker declaration variadic and appends the tolerance pair,
; which is what the front end emits for `poseidon_tau, <value>`.
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-04) #5/' %s > %t.tau4.ll
; RUN: rm -rf %t.c4 && %opt %t.tau4.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t.c4 -poseidon-print -S 2>&1 | FileCheck --check-prefix=SITE4 %s
;
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-07) #5/' %s > %t.tau7.ll
; RUN: rm -rf %t.c7 && %opt %t.tau7.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t.c7 -poseidon-print -S 2>&1 | FileCheck --check-prefix=SITE7 %s
;
; A tolerance nothing on the frontier meets refuses the site and keeps FP64.
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-08) #5/' %s > %t.tau8.ll
; RUN: rm -rf %t.c8 && %opt %t.tau8.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t.c8 -poseidon-print -S 2>&1 | FileCheck --check-prefix=SITE8 %s
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

; CHECK: [poseidon] Found 1 AbstractMatmul(s) for preprocess_matmul_body
; CHECK-NEXT: Matmul[0]: 16x16x16 a=f64 b=f64 acc=f64 d=f64 origin=ScalarLoopReduction
; CHECK: matmul[0] pricing: baseline/MAC=4.988291e+04 executions=8192
; CHECK-NEXT: #0 wmma m16n16k16 f16/f32 compCost/MAC=3.385254e+03 rel=6.786400e-02
; CHECK-NEXT: #1 tcec n=2 wmma m16n16k16 f16/f32 compCost/MAC=4.159686e+03 rel=8.338900e-02
; CHECK-NEXT: #2 tcec n=3 wmma m16n16k16 f16/f32 compCost/MAC=4.899349e+03 rel=9.821700e-02
; CHECK: Matmul[0] candidates (initial cost=4.988291e+04, executions=8192,
; CHECK-NEXT: wmma m16n16k16 f16/f32: {{.*}}cost=-380908769
; CHECK-NEXT: tcec n=2 wmma m16n16k16 f16/f32: {{.*}}cost=-374564621
; CHECK-NEXT: tcec n=3 wmma m16n16k16 f16/f32: {{.*}}cost=-368505296
; CHECK: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; CHECK-LABEL: define {{.*}} @preprocess_matmul_body(
; CHECK: fptrunc double %{{.+}} to float
; CHECK: fptrunc float %{{.+}} to half
; CHECK: @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(
; CHECK: @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(
; CHECK: @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(
; CHECK: @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(
; CHECK: ret void

; CLEAN-LABEL: define {{.*}} @preprocess_matmul_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN: ret void

; SITE4: Poseidon error-budget: matmul -> tcec n=2 wmma m16n16k16 f16/f32 (domain err 1.572576e-07 <= 1.000000e-04 at confidence 9.500000e-01)
; SITE4: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32

; SITE7: Poseidon error-budget: matmul -> tcec n=3 wmma m16n16k16 f16/f32 (domain err 5.874975e-08 <= 1.000000e-07 at confidence 9.500000e-01)
; SITE7: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32

; SITE8: Poseidon error-budget: matmul -> baseline F64 (no rewrite clears 1.000000e-08 at confidence 9.500000e-01 faster than F64)
; SITE8: no rewrite applied for preprocess_matmul_body; calling original body matmul_body
; SITE8-NOT: Applying solution for matmul[0]
