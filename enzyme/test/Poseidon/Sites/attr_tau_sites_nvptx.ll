; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/annotated_kernel/preprocess_fms_kernel_poseidon_body.fpprofile %t/prof/preprocess_fms_loose_poseidon_body.fpprofile
; RUN: cp %S/../Inputs/annotated_kernel/preprocess_fms_kernel_poseidon_body.fpprofile %t/prof/preprocess_fms_tight_poseidon_body.fpprofile
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/cache -S > %t/out.ll 2> %t/err.txt
; RUN: FileCheck --check-prefix=ERR %s < %t/err.txt
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll
; RUN: grep -B3 -A4 '%ds.hi64[^ ]* = fpext' %t/out.ll | FileCheck --check-prefix=JOIN %s

; Three sites in one module, same body and same profile, differing only in what
; the annotation asks for: POSEIDON_OPTIMIZE_TAU(1e-7) takes FP32,
; POSEIDON_OPTIMIZE_TAU(1e-12) takes the two-component expansion, and bare
; POSEIDON_OPTIMIZE with no profile of its own is left exactly as written
; instead of failing the compile.

; ModuleID = 'attr_tau.cu'
source_filename = "attr_tau.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.file = private unnamed_addr constant [12 x i8] c"attr_tau.cu\00", section "llvm.metadata"
@.loose = private unnamed_addr constant { double } { double 1.000000e-07 }, section "llvm.metadata"
@.tight = private unnamed_addr constant { double } { double 1.000000e-12 }, section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @fms_loose, ptr @.str, ptr @.file, i32 2, ptr @.loose },
  { ptr, ptr, ptr, i32, ptr } { ptr @fms_tight, ptr @.str, ptr @.file, i32 3, ptr @.tight },
  { ptr, ptr, ptr, i32, ptr } { ptr @fms_bare, ptr @.str, ptr @.file, i32 4, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @fms_loose(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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

define dso_local ptx_kernel void @fms_tight(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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

define dso_local ptx_kernel void @fms_bare(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

; The annotation alone must not change the kernel: with no profile the site is
; left as written (no outline, no call), and with a profile whose solve applies
; nothing the outline is folded back. With a rewrite applied the materialized
; body is folded back the same way. In every case the kernel is one entry
; function with the arithmetic inline and no _poseidon_body symbol survives.

; ERR: [poseidon] preprocess_fms_loose_poseidon_body: tau=1.000000e-07 {{.*}} rel=1.428768e-08
; ERR: Applying solution for CS: All FP64(0%) + FP32(100%)
; ERR: [poseidon] pin: fms_loose:PT0=All FP64(0%) + FP32(100%)
; ERR: [poseidon] preprocess_fms_tight_poseidon_body: tau=1.000000e-12 {{.*}} rel=7.011882e-16
; ERR: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; ERR: [poseidon] pin: fms_tight:PT0=All FP64(0%) + Expansion2(100%)
; ERR: warning: <unknown>:0:0: in function fms_bare void (ptr, ptr, ptr): Poseidon: no profile at {{.*}}preprocess_fms_bare_poseidon_body.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; ERR: [poseidon] fms_bare_poseidon_body: the profiled program did not contain this site (no profile at {{.*}}preprocess_fms_bare_poseidon_body.fpprofile); left unchanged
; ERR-NOT: Applying solution

; The loose site runs the body in FP32 between one narrowing of each input and
; one widening of the result.
; IR-LABEL: define {{.*}} @fms_loose(
; IR-SAME: ptr {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[B:%[0-9]+]], ptr {{.*}}[[OUT:%[0-9]+]])
; IR: [[I:%[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; IR-NEXT: [[AP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[A]], i64 [[I]]
; IR-NEXT: [[AV:%[0-9]+]] = load double, ptr [[AP]]
; IR-NEXT: [[BP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[B]], i64 [[I]]
; IR-NEXT: [[BV:%[0-9]+]] = load double, ptr [[BP]]
; IR-NEXT: [[AF:%[^ ]+]] = fptrunc double [[AV]] to float
; IR-NEXT: [[BF:%[^ ]+]] = fptrunc double [[BV]] to float
; IR-NEXT: [[T0:%[0-9]+]] = call float @llvm.fmuladd.f32(float [[AF]], float [[BF]], float [[AF]])
; IR-NEXT: [[NB:%[0-9]+]] = fneg float [[BF]]
; IR-NEXT: [[T1:%[0-9]+]] = call float @llvm.fmuladd.f32(float [[T0]], float [[T0]], float [[NB]])
; IR-NEXT: [[T2:%[0-9]+]] = call float @llvm.fmuladd.f32(float [[T1]], float [[BF]], float [[T0]])
; IR-NEXT: [[NA:%[0-9]+]] = fneg float [[AF]]
; IR-NEXT: [[T3:%[0-9]+]] = call float @llvm.fmuladd.f32(float [[T2]], float [[T2]], float [[NA]])
; IR-NEXT: [[T4:%[0-9]+]] = call float @llvm.fmuladd.f32(float [[T3]], float [[T1]], float [[T2]])
; IR-NEXT: [[R:%[0-9]+]] = fpext float [[T4]] to double
; IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[OUT]], i64 [[I]]
; IR-NEXT: store double [[R]], ptr [[OP]]
; IR-NEXT: ret void

; The tight site splits each input into a float pair, carries pairs through
; float arithmetic alone, and stores the sum of the last pair.
; IR-LABEL: define {{.*}} @fms_tight(
; IR-SAME: ptr {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[B:%[0-9]+]], ptr {{.*}}[[OUT:%[0-9]+]])
; IR: [[I:%[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; IR-NEXT: [[AP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[A]], i64 [[I]]
; IR-NEXT: [[AV:%[0-9]+]] = load double, ptr [[AP]]
; IR-NEXT: [[BP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[B]], i64 [[I]]
; IR-NEXT: [[BV:%[0-9]+]] = load double, ptr [[BP]]
; IR-NEXT: [[AHI:%[^ ]+]] = fptrunc double [[AV]] to float
; IR-NEXT: [[AHID:%[^ ]+]] = fpext float [[AHI]] to double
; IR-NEXT: [[ALOD:%[^ ]+]] = fsub double [[AV]], [[AHID]]
; IR-NEXT: [[ALO:%[^ ]+]] = fptrunc double [[ALOD]] to float
; IR-NEXT: [[BHI:%[^ ]+]] = fptrunc double [[BV]] to float
; IR-NEXT: [[BHID:%[^ ]+]] = fpext float [[BHI]] to double
; IR-NEXT: [[BLOD:%[^ ]+]] = fsub double [[BV]], [[BHID]]
; IR-NEXT: [[BLO:%[^ ]+]] = fptrunc double [[BLOD]] to float
; IR-NEXT: [[P:%[^ ]+]] = fmul float [[AHI]], [[BHI]]
; IR-NEXT: [[NP:%[^ ]+]] = fneg float [[P]]
; IR-NEXT: [[E0:%[^ ]+]] = call float @llvm.fma.f32(float [[AHI]], float [[BHI]], float [[NP]])
; IR-NEXT: [[E1:%[^ ]+]] = call float @llvm.fma.f32(float [[AHI]], float [[BLO]], float [[E0]])
; IR-NEXT: call float @llvm.fma.f32(float [[ALO]], float [[BHI]], float [[E1]])
; IR-NOT: double
; IR: [[HI:%[^ ]+]] = fpext float %{{[^ ]+}} to double
; IR-NEXT: [[LO:%[^ ]+]] = fpext float %{{[^ ]+}} to double
; IR-NEXT: [[J:%[^ ]+]] = fadd double [[HI]], [[LO]], !poseidon.ds.join !{{[0-9]+}}
; IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[OUT]], i64 [[I]]
; IR-NEXT: store double [[J]], ptr [[OP]]
; IR-NEXT: ret void

; The joined pair is the last fast two-sum of the expansion.
; JOIN: [[S:%[^ ]+]] = fadd float [[X:%[^ ,]+]], [[Y:%[^ ,]+]]{{$}}
; JOIN-NEXT: [[BP:%[^ ]+]] = fsub float [[S]], [[X]]{{$}}
; JOIN-NEXT: [[E:%[^ ]+]] = fsub float [[Y]], [[BP]]{{$}}
; JOIN-NEXT: [[HI:%[^ ]+]] = fpext float [[S]] to double
; JOIN-NEXT: [[LO:%[^ ]+]] = fpext float [[E]] to double
; JOIN-NEXT: [[J:%[^ ]+]] = fadd double [[HI]], [[LO]], !poseidon.ds.join !{{[0-9]+}}
; JOIN-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}, i64 %{{[0-9]+}}
; JOIN-NEXT: store double [[J]], ptr [[OP]]

; An unprofiled site keeps its FP64 arithmetic and leaves no outlined body
; behind.
; IR-LABEL: define {{.*}} @fms_bare(
; IR-SAME: ptr {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[B:%[0-9]+]], ptr {{.*}}[[OUT:%[0-9]+]])
; IR-NOT: call void @
; IR: [[I:%[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; IR-NEXT: [[AP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[A]], i64 [[I]]
; IR-NEXT: [[AV:%[0-9]+]] = load double, ptr [[AP]]
; IR-NEXT: [[BP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[B]], i64 [[I]]
; IR-NEXT: [[BV:%[0-9]+]] = load double, ptr [[BP]]
; IR-NEXT: [[T0:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[AV]], double [[BV]], double [[AV]])
; IR-NEXT: [[NB:%[0-9]+]] = fneg double [[BV]]
; IR-NEXT: [[T1:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T0]], double [[T0]], double [[NB]])
; IR-NEXT: [[T2:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T1]], double [[BV]], double [[T0]])
; IR-NEXT: [[NA:%[0-9]+]] = fneg double [[AV]]
; IR-NEXT: [[T3:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T2]], double [[T2]], double [[NA]])
; IR-NEXT: [[T4:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T3]], double [[T1]], double [[T2]])
; IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[OUT]], i64 [[I]]
; IR-NEXT: store double [[T4]], ptr [[OP]]
; IR-NEXT: ret void
; IR-NOT: _poseidon_body
