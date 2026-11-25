; DEFINE: %{opt} = %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 \
; DEFINE:   -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/ds_expansion/preprocess_fms_body.fpprofile %t/prof/preprocess_fms_kernel_poseidon_body.fpprofile

; The accuracy target of the site below, the kernel and profile of
; Sites/annotated_kernel_nvptx.ll.

; A site tolerance is a per-operation relative rounding level: the solve takes
; the cheapest frontier point whose modelled error, put back on that scale,
; clears it. The site is solved at three tolerances against one cached DP
; table, so the second and third compiles read the baseline back out of the
; table. The site's own FP64 body sits at A0/S = 2.16e-17, so the third
; tolerance is refused rather than met with a made-up number, and the site
; keeps the original body bit for bit.
; RUN: mkdir %t/ladder
; RUN: sed "s/double 1.000000e-07/double 1.000000e-12/" %s > %t/ladder/mid.ll
; RUN: sed "s/double 1.000000e-07/double 1.000000e-18/" %s > %t/ladder/tight.ll
; RUN: %opt %s %{opt} -poseidon-cache=%t/ladder/c -poseidon-explain=%t/ladder/loose 2>&1 | tee %t/ladder/loose.log | FileCheck --check-prefix=LADDER-LOOSE %s
; RUN: %opt %t/ladder/mid.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefixes=LADDER-MID,EXP2 --implicit-check-not="Solving the DP" %s
; RUN: %opt %t/ladder/tight.ll %{opt} -poseidon-cache=%t/ladder/c -poseidon-explain=%t/ladder/tight 2>&1 | tee %t/ladder/tight.log | FileCheck --check-prefix=LADDER-TIGHT --implicit-check-not="Solving the DP" --implicit-check-not="!poseidon.ds.join" %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/ladder/loose.log %t/ladder/loose | FileCheck --check-prefix=LOOSE-JSON %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/ladder/tight.log %t/ladder/tight | FileCheck --check-prefix=TIGHT-JSON %s
; RUN: FileCheck --check-prefix=TIGHT-BEST %s < %t/ladder/tight/preprocess_fms_kernel_poseidon_body.explain.json

; LADDER-LOOSE: Per-operation relative error tolerance: 1.000000e-07
; LADDER-LOOSE-NEXT: Solving the DP for preprocess_fms_kernel_poseidon_body.
; LADDER-LOOSE: DP table for preprocess_fms_kernel_poseidon_body cached to
; LADDER-LOOSE: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-07 S=1.187943e+04 A0=2.571104e-13 selected cost=-9.223869e-01 accCost=1.697296e-04 rel=1.428768e-08
; LADDER-LOOSE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; LADDER-LOOSE: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + FP32(100%)
; LADDER-LOOSE-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LADDER-LOOSE: call float @llvm.fmuladd.f32(
; LADDER-LOOSE-NOT: !poseidon.ds.join

; LADDER-MID: Per-operation relative error tolerance: 1.000000e-12
; LADDER-MID-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-MID: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost=-1.692668e-01 accCost=8.072609e-12 rel=7.011882e-16
; LADDER-MID: Applying solution for CS: All FP64(0%) + Expansion2(100%) (#2)
; LADDER-MID: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + Expansion2(100%)

; The explanation of each compile says what its decision lines say.
; LOOSE-JSON: preprocess_fms_kernel_poseidon_body: rewritten{{$}}
; TIGHT-JSON: preprocess_fms_kernel_poseidon_body: no_solution; no frontier point meets the accuracy target 1e-18{{$}}
; TIGHT-BEST: "best_achievable_rel_error": 2.16433{{[0-9]+}}e-17,

; LADDER-TIGHT: Per-operation relative error tolerance: 1.000000e-18
; LADDER-TIGHT-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-TIGHT: No solution found that meets accuracy tolerance 1.000000e-18!
; LADDER-TIGHT-NEXT: Best achievable relative accuracy in DP table: 2.164332e-17
; LADDER-TIGHT: [poseidon] pin: {{$}}
; LADDER-TIGHT: warning: {{.*}}Poseidon: Poseidon returned false (no change) for preprocess_fms_kernel_poseidon_body
; LADDER-TIGHT: [poseidon] no rewrite applied for preprocess_fms_kernel_poseidon_body; calling original body fms_kernel_poseidon_body (bit-transparent no-op)
; LADDER-TIGHT-LABEL: define {{.*}} @fms_kernel(
; LADDER-TIGHT-SAME: ptr {{.*}}[[KA:%[0-9]+]], ptr {{.*}}[[KB:%[0-9]+]], ptr {{.*}}[[KC:%[0-9]+]])
; LADDER-TIGHT-NEXT: entry:
; LADDER-TIGHT-NEXT: call void @fms_kernel_poseidon_body(ptr [[KA]], ptr [[KB]], ptr [[KC]])
; LADDER-TIGHT-NEXT: ret void
; LADDER-TIGHT-LABEL: define internal void @fms_kernel_poseidon_body(
; LADDER-TIGHT-COUNT-5: call double @llvm.fmuladd.f64(
; LADDER-TIGHT-NOT: preprocess_
; LADDER-TIGHT-NOT: !poseidon.ds.join

; Expansion2 joins the result pair into the stored double.
; EXP2-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; EXP2: fadd double %{{.+}}, %{{.+}}, !poseidon.ds.join

source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [14 x i8] c"tau_ladder.cu\00", section "llvm.metadata"
@.args = private unnamed_addr constant { double } { double 1.000000e-07 }, section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr @.args }], section "llvm.metadata"

define dso_local ptx_kernel void @fms_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
