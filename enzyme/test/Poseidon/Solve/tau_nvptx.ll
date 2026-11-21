; DEFINE: %{opt} = %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/annotated_kernel -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 \
; DEFINE:   -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -S
; RUN: rm -rf %t && mkdir %t

; The accuracy target of the site below, the kernel and profile of
; annotated_kernel_nvptx.ll.

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
; RUN: %opt %s %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefix=LADDER-LOOSE %s
; RUN: %opt %t/ladder/mid.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefix=LADDER-MID %s
; RUN: %opt %t/ladder/tight.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefix=LADDER-TIGHT %s

; LADDER-LOOSE: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-07 S=1.187943e+04 A0=2.571104e-13 selected cost={{-[0-9.e+-]+}} accCost={{.*}} rel=1.428768e-08
; LADDER-LOOSE: Applying solution for CS: All FP64(0%) + FP32(100%)
; LADDER-LOOSE-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LADDER-LOOSE: fptrunc double
; LADDER-LOOSE-NOT: !poseidon.ds.join

; LADDER-MID: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost={{-[0-9.e+-]+}} accCost={{.*}} rel=7.011882e-16
; LADDER-MID: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; LADDER-MID-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LADDER-MID: !poseidon.ds.join

; LADDER-TIGHT: No solution found that meets accuracy tolerance 1.000000e-18!
; LADDER-TIGHT: Best achievable relative accuracy in DP table: 2.164332e-17
; LADDER-TIGHT: [poseidon] no rewrite applied for preprocess_fms_kernel_poseidon_body
; LADDER-TIGHT-LABEL: define {{.*}} @fms_kernel(
; LADDER-TIGHT: call double @llvm.fmuladd.f64
; LADDER-TIGHT-NOT: !poseidon.ds.join

; -poseidon-tau on the command line is the accuracy target of a site that
; carries none, and a site with no matrix product solves its elementwise work
; under it, as the MID tolerance written at the site does.
; RUN: mkdir %t/flag
; RUN: sed 's/ptr @.args }/ptr null }/' %s > %t/flag/flag.ll
; RUN: FileCheck --check-prefix=FLAG-EDIT %s < %t/flag/flag.ll
; RUN: %opt %t/flag/flag.ll %{opt} -poseidon-cache=%t/flag/c -poseidon-tau=1e-12 2>&1 | FileCheck --check-prefix=FLAG %s

; FLAG-EDIT: {{^}}@llvm.global.annotations = {{.*}} i32 2, ptr null }]

; FLAG: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost={{-[0-9.e+-]+}} accCost={{.*}} rel=7.011882e-16
; FLAG: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; FLAG-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; FLAG: !poseidon.ds.join

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
