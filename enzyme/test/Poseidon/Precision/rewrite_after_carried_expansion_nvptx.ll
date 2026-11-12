; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/rewrite_after_carried_expansion %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/rewrite_after_carried_expansion -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/rewrite_after_carried_expansion/fake-herbie.sh -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv "-poseidon-apply-rewrites=PT0=All FP64(0%%) + Expansion2(100%%),R2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2))" -poseidon-cache=%t/expansion-first -S -o %t/expansion-first.ll 2> %t/expansion-first.err
; RUN: FileCheck --check-prefix=APPLY %s < %t/expansion-first.err
; RUN: FileCheck --check-prefix=IR %s < %t/expansion-first.ll
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/rewrite_after_carried_expansion %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/rewrite_after_carried_expansion -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/rewrite_after_carried_expansion/fake-herbie.sh -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv "-poseidon-apply-rewrites=R2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2)),PT0=All FP64(0%%) + Expansion2(100%%)" -poseidon-cache=%t/rewrite-first -S -o %t/rewrite-first.ll
; RUN: FileCheck --check-prefix=IR %s < %t/rewrite-first.ll

; A two-component expansion of one FP subgraph, then a rewrite of another that
; reads the FP64 accumulator the expansion carries as a pair. In the loop of
; carried_body (Inputs/rewrite_after_carried_expansion/carried.cu) the
; accumulator %13 feeds the subgraph {%22 %23} that updates it (PT0) and the
; subgraph {%14 %15 %16} that stores (acc - y) * (y + acc) (R2, with Herbie's
; answer canned next to the profile). PT0 carries %13 as two float PHIs and
; erases it, its users reading the pair joined back to binary64; the rewrite,
; which names %13 as v1, reads that same value. Applied in either order, the
; two steps emit one program up to value names.
; The module: clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -S -emit-llvm carried.cu. Profile: the same source through
; poseidon-clang++ -poseidon-profile-generate, run on an RTX 5090.

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
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
  store double %16, ptr %19, align 8, !tbaa !10
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %18
  %21 = load double, ptr %20, align 8, !tbaa !10
  %22 = fmul double %21, %21
  %23 = tail call double @llvm.fmuladd.f64(double %13, double 5.000000e-01, double %22)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %10, label %11, !llvm.loop !12
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z14carried_kernelPKddPdS1_(ptr noundef %0, double noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !15
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !15
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef nonnull @carried_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, double noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git 9d618fc0243103263ebf9bdcd08ef25e871d9c78)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
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

; APPLY: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; APPLY-NEXT: Applying solution for (* (- v1 v2) (+ v2 v1)) --(0)-> (-.f64 (*.f64 v1 v1) (*.f64 v2 v2))

; IR-LABEL: define dso_local void @preprocess_carried_body(
; IR-NOT: phi double
; IR: %ds.phi.hi = phi float
; IR-NEXT: %ds.phi.lo = phi float
; IR-NEXT: %[[HI:.+]] = fpext float %ds.phi.hi to double
; IR-NEXT: %[[LO:.+]] = fpext float %ds.phi.lo to double
; IR-NEXT: %[[ACC:.+]] = fadd double %[[HI]], %[[LO]]
; IR-NOT: phi double
; IR: %herbie.mul = fmul double %[[ACC]], %[[ACC]]
; IR-NEXT: %herbie.sub = fsub double %herbie.mul, %herbie.mul{{[0-9]+}}
; IR-NOT: phi double
; IR: {{^}}}{{$}}
