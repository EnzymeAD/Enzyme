; DEFINE: %{opt} = %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
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
; RUN: %opt %t/ladder/mid.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefixes=LADDER-MID,EXP2 --implicit-check-not="Solving the DP" %s
; RUN: %opt %t/ladder/tight.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefix=LADDER-TIGHT --implicit-check-not="Solving the DP" --implicit-check-not="!poseidon.ds.join" %s

; LADDER-LOOSE: Per-operation relative error tolerance: 1.000000e-07
; LADDER-LOOSE-NEXT: Solving the DP for preprocess_fms_kernel_poseidon_body.
; LADDER-LOOSE: DP table for preprocess_fms_kernel_poseidon_body cached to file.
; LADDER-LOOSE: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-07 S=1.187943e+04 A0=2.571104e-13 selected cost=-9.223869e-01 accCost=1.697296e-04 rel=1.428768e-08
; LADDER-LOOSE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; LADDER-LOOSE: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + FP32(100%)
; LADDER-LOOSE-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LADDER-LOOSE-SAME: ptr {{.*}}[[PA:%[0-9]+]], ptr {{.*}}[[PB:%[0-9]+]], ptr {{.*}}[[PC:%[0-9]+]])
; LADDER-LOOSE: [[AP:%.+]] = getelementptr {{.*}}ptr [[PA]], i64 [[I:%.+]]
; LADDER-LOOSE-NEXT: [[A:%.+]] = load double, ptr [[AP]],
; LADDER-LOOSE-NEXT: [[BP:%.+]] = getelementptr {{.*}}ptr [[PB]], i64 [[I]]{{$}}
; LADDER-LOOSE-NEXT: [[B:%.+]] = load double, ptr [[BP]],
; LADDER-LOOSE-NEXT: [[AF:%.+]] = fptrunc double [[A]] to float
; LADDER-LOOSE-NEXT: [[BF:%.+]] = fptrunc double [[B]] to float
; LADDER-LOOSE-NEXT: [[V0:%.+]] = tail call float @llvm.fmuladd.f32(float [[AF]], float [[BF]], float [[AF]])
; LADDER-LOOSE-NEXT: [[NB:%.+]] = fneg float [[BF]]{{$}}
; LADDER-LOOSE-NEXT: [[V1:%.+]] = tail call float @llvm.fmuladd.f32(float [[V0]], float [[V0]], float [[NB]])
; LADDER-LOOSE-NEXT: [[V2:%.+]] = tail call float @llvm.fmuladd.f32(float [[V1]], float [[BF]], float [[V0]])
; LADDER-LOOSE-NEXT: [[NA:%.+]] = fneg float [[AF]]{{$}}
; LADDER-LOOSE-NEXT: [[V3:%.+]] = tail call float @llvm.fmuladd.f32(float [[V2]], float [[V2]], float [[NA]])
; LADDER-LOOSE-NEXT: [[V4:%.+]] = tail call float @llvm.fmuladd.f32(float [[V3]], float [[V1]], float [[V2]])
; LADDER-LOOSE-NEXT: [[R:%.+]] = fpext float [[V4]] to double
; LADDER-LOOSE-NEXT: [[RP:%.+]] = getelementptr {{.*}}ptr [[PC]], i64 [[I]]{{$}}
; LADDER-LOOSE-NEXT: store double [[R]], ptr [[RP]],
; LADDER-LOOSE-NOT: !poseidon.ds.join

; LADDER-MID: Per-operation relative error tolerance: 1.000000e-12
; LADDER-MID-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-MID: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost=-1.692668e-01 accCost=8.072609e-12 rel=7.011882e-16
; LADDER-MID: Applying solution for CS: All FP64(0%) + Expansion2(100%) (#2)
; LADDER-MID: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + Expansion2(100%)

; LADDER-TIGHT: Per-operation relative error tolerance: 1.000000e-18
; LADDER-TIGHT-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-TIGHT: No solution found that meets accuracy tolerance 1.000000e-18!
; LADDER-TIGHT-NEXT: Best achievable relative accuracy in DP table: 2.164332e-17
; LADDER-TIGHT-NEXT: [poseidon] pin: {{$}}
; LADDER-TIGHT: warning: {{.*}}Poseidon: Poseidon returned false (no change) for preprocess_fms_kernel_poseidon_body
; LADDER-TIGHT: [poseidon] no rewrite applied for preprocess_fms_kernel_poseidon_body; calling original body fms_kernel_poseidon_body (bit-transparent no-op)
; LADDER-TIGHT-LABEL: define {{.*}} @fms_kernel(
; LADDER-TIGHT-SAME: ptr {{.*}}[[KA:%[0-9]+]], ptr {{.*}}[[KB:%[0-9]+]], ptr {{.*}}[[KC:%[0-9]+]])
; LADDER-TIGHT-NEXT: entry:
; LADDER-TIGHT-NEXT: call void @fms_kernel_poseidon_body(ptr [[KA]], ptr [[KB]], ptr [[KC]])
; LADDER-TIGHT-NEXT: ret void
; LADDER-TIGHT-LABEL: define internal void @fms_kernel_poseidon_body(
; LADDER-TIGHT-SAME: ptr {{.*}}[[PA:%[0-9]+]], ptr {{.*}}[[PB:%[0-9]+]], ptr {{.*}}[[PC:%[0-9]+]])
; LADDER-TIGHT: [[AP:%.+]] = getelementptr {{.*}}ptr [[PA]], i64 [[I:%.+]]
; LADDER-TIGHT-NEXT: [[A:%.+]] = load double, ptr [[AP]],
; LADDER-TIGHT-NEXT: [[BP:%.+]] = getelementptr {{.*}}ptr [[PB]], i64 [[I]]{{$}}
; LADDER-TIGHT-NEXT: [[B:%.+]] = load double, ptr [[BP]],
; LADDER-TIGHT-NEXT: [[V0:%.+]] = tail call double @llvm.fmuladd.f64(double [[A]], double [[B]], double [[A]])
; LADDER-TIGHT-NEXT: [[NB:%.+]] = fneg double [[B]]{{$}}
; LADDER-TIGHT-NEXT: [[V1:%.+]] = tail call double @llvm.fmuladd.f64(double [[V0]], double [[V0]], double [[NB]])
; LADDER-TIGHT-NEXT: [[V2:%.+]] = tail call double @llvm.fmuladd.f64(double [[V1]], double [[B]], double [[V0]])
; LADDER-TIGHT-NEXT: [[NA:%.+]] = fneg double [[A]]{{$}}
; LADDER-TIGHT-NEXT: [[V3:%.+]] = tail call double @llvm.fmuladd.f64(double [[V2]], double [[V2]], double [[NA]])
; LADDER-TIGHT-NEXT: [[V4:%.+]] = tail call double @llvm.fmuladd.f64(double [[V3]], double [[V1]], double [[V2]])
; LADDER-TIGHT-NEXT: [[RP:%.+]] = getelementptr {{.*}}ptr [[PC]], i64 [[I]]{{$}}
; LADDER-TIGHT-NEXT: store double [[V4]], ptr [[RP]],
; LADDER-TIGHT-NEXT: ret void
; LADDER-TIGHT-NOT: preprocess_
; LADDER-TIGHT-NOT: !poseidon.ds.join

; -poseidon-tau on the command line is the accuracy target of a site that
; carries none, and a site with no matrix product solves its elementwise work
; under it, as the MID tolerance written at the site does.
; RUN: mkdir %t/flag
; RUN: sed 's/ptr @.args }/ptr null }/' %s > %t/flag/flag.ll
; RUN: FileCheck --check-prefix=FLAG-EDIT %s < %t/flag/flag.ll
; RUN: %opt %t/flag/flag.ll %{opt} -poseidon-cache=%t/flag/c -poseidon-tau=1e-12 2>&1 | FileCheck --check-prefixes=FLAG,EXP2 %s

; FLAG-EDIT: {{^}}@llvm.global.annotations = {{.*}} i32 2, ptr null }]

; FLAG: Per-operation relative error tolerance: 1.000000e-12
; FLAG: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost=-1.692668e-01 accCost=8.072609e-12 rel=7.011882e-16
; FLAG: Applying solution for CS: All FP64(0%) + Expansion2(100%) (#2)
; FLAG: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + Expansion2(100%)

; Expansion2 splits each loaded double into a float pair, computes in float
; only, and joins the result pair into the stored double.
; EXP2-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; EXP2-SAME: ptr {{.*}}[[PA:%[0-9]+]], ptr {{.*}}[[PB:%[0-9]+]], ptr {{.*}}[[PC:%[0-9]+]])
; EXP2: [[AP:%.+]] = getelementptr {{.*}}ptr [[PA]], i64 [[I:%.+]]
; EXP2-NEXT: [[A:%.+]] = load double, ptr [[AP]],
; EXP2-NEXT: [[BP:%.+]] = getelementptr {{.*}}ptr [[PB]], i64 [[I]]{{$}}
; EXP2-NEXT: [[B:%.+]] = load double, ptr [[BP]],
; EXP2-NEXT: [[AH:%.+]] = fptrunc double [[A]] to float
; EXP2-NEXT: [[AHD:%.+]] = fpext float [[AH]] to double
; EXP2-NEXT: [[ALD:%.+]] = fsub double [[A]], [[AHD]]{{$}}
; EXP2-NEXT: [[AL:%.+]] = fptrunc double [[ALD]] to float
; EXP2-NEXT: [[BH:%.+]] = fptrunc double [[B]] to float
; EXP2-NEXT: [[BHD:%.+]] = fpext float [[BH]] to double
; EXP2-NEXT: [[BLD:%.+]] = fsub double [[B]], [[BHD]]{{$}}
; EXP2-NEXT: [[BL:%.+]] = fptrunc double [[BLD]] to float
; EXP2-NEXT: [[P:%.+]] = fmul float [[AH]], [[BH]]{{$}}
; EXP2-NEXT: [[NP:%.+]] = fneg float [[P]]{{$}}
; EXP2-NEXT: [[E0:%.+]] = tail call float @llvm.fma.f32(float [[AH]], float [[BH]], float [[NP]])
; EXP2-NEXT: [[E1:%.+]] = tail call float @llvm.fma.f32(float [[AH]], float [[BL]], float [[E0]])
; EXP2-NEXT: {{%.+}} = tail call float @llvm.fma.f32(float [[AL]], float [[BH]], float [[E1]])
; EXP2-NOT: double
; EXP2: [[HI:%.+]] = fpext float {{%.+}} to double
; EXP2-NEXT: [[LO:%.+]] = fpext float {{%.+}} to double
; EXP2-NEXT: [[J:%.+]] = fadd double [[HI]], [[LO]], !poseidon.ds.join
; EXP2-NEXT: [[RP:%.+]] = getelementptr {{.*}}ptr [[PC]], i64 [[I]]{{$}}
; EXP2-NEXT: store double [[J]], ptr [[RP]],
; EXP2-NEXT: ret void

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
