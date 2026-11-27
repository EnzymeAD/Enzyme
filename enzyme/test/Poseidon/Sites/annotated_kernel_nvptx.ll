; DEFINE: %{use} = -poseidon-profile-use=%t/prof -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; RUN: rm -rf %t && split-file %s %t && mkdir %t/prof %t/tauprof
; RUN: sed 's/@KERNEL/@fms_kernel/' %t/body.ll | cat %t/kernel.ll - %t/decls.ll > %t/k.ll
; RUN: for k in fms_loose fms_tight fms_bare; do sed "s/@KERNEL/@$k/" %t/body.ll; done | cat %t/tau.ll - %t/decls.ll > %t/tau3.ll
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/tauprof/fms_loose.fpprofile
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/tauprof/fms_tight.fpprofile
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/prof/fms_kernel.fpprofile
; RUN: %opt %t/k.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" %{use} -poseidon-cache=%t/attr -poseidon-cost-budget=-1e-9 -poseidon-print -S > %t/attr.out 2>&1
; RUN: FileCheck %s < %t/attr.out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/attr.out
; RUN: sed -n '/^; ModuleID/,$p' %t/attr.out > %t/attr.ll
; RUN: FileCheck --check-prefixes=MODULE,BODY %s < %t/attr.ll
; RUN: %opt %t/k.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-kernels=fms.* %{use} -poseidon-cache=%t/regex -poseidon-cost-budget=-1e-9 -poseidon-print -S 2>&1 | FileCheck --check-prefix=REGEX %s
; RUN: %opt %t/k.ll %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -S 2> %t/noprofile.err | FileCheck --check-prefixes=NOPROFILE,FP64 %s
; RUN: FileCheck --check-prefix=NOPROFILE-WARN %s < %t/noprofile.err
; RUN: %opt %t/k.ll %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/noop -poseidon-cost-budget=0 -S 2> %t/noop.err | FileCheck --check-prefixes=NOOP,FP64 %s
; RUN: %if nvptx-registered-target %{ count 0 < %t/noop.err %}
; RUN: %opt %t/k.ll %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/inlined -poseidon-cost-budget=-1e-9 -S 2> %t/inlined.err | FileCheck --check-prefix=INLINED %s
; RUN: %if nvptx-registered-target %{ count 0 < %t/inlined.err %}
; RUN: %opt %t/tau3.ll %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%t/tauprof -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/tau -S -poseidon-print > %t/tau.out 2> %t/tau.err
; RUN: FileCheck --check-prefix=TAU-ERR %s < %t/tau.err
; RUN: FileCheck --check-prefix=TAU-IR %s < %t/tau.out

; A kernel carrying POSEIDON_OPTIMIZE is a site whose computation is its whole
; body, with no marker call and no shadow arguments (same code and profile as
; expansion_two_components_nvptx.ll), also when named by -poseidon-kernels. A
; run that applies nothing or inlines prints nothing, checked only where LLVM
; has an NVPTX target (opt warns that it cannot create the target machine).

; CHECK: Candidates:
; CHECK: All FP64(0%) + FP32(100%)
; CHECK: All FP64(0%) + Expansion2(100%)
; CHECK: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; CHECK: [poseidon] pin: fms_kernel:P0=All FP64(0%) + Expansion2(100%)

; Without inlining the kernel calls the rewritten body with its own arguments.
; MODULE-LABEL: define {{.*}} @fms_kernel(
; MODULE-SAME: ptr {{.*}}%[[KA:[0-9]+]], ptr {{.*}}%[[KB:[0-9]+]], ptr {{.*}}%[[KOUT:[0-9]+]])
; MODULE-NEXT: entry:
; MODULE-NEXT: call void @preprocess_fms_kernel_poseidon_body(ptr %[[KA]], ptr %[[KB]], ptr %[[KOUT]])
; MODULE-NEXT: ret void

; The body is the double-single expansion (expansion_two_components_nvptx.ll
; checks its instructions), joined back to FP64 at the store.
; BODY-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; BODY-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; BODY: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; BODY: fptrunc double %{{[0-9]+}} to float
; BODY: %[[J:pair.f64[0-9]*]] = fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join
; BODY-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; BODY-NEXT: store double %[[J]], ptr %[[OP]]
; BODY-NEXT: ret void

; CLEAN-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN-NOT: fmul double
; CLEAN-NOT: fneg double
; CLEAN: ret void

; REGEX: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; REGEX: [poseidon] pin: fms_kernel:P0=All FP64(0%) + Expansion2(100%)
; REGEX-LABEL: define {{.*}} @fms_kernel(
; REGEX: call void @preprocess_fms_kernel_poseidon_body(

; The annotation alone does not change the kernel: unprofiled, applying
; nothing or inlining the rewritten body, the kernel is one function with the
; arithmetic inline and no _poseidon_body symbol survives.
; NOPROFILE-WARN: warning: <unknown>:0:0: in function fms_kernel void (ptr, ptr, ptr): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; NOPROFILE-WARN-NOT: warning
; NOPROFILE-LABEL: define {{.*}} @fms_kernel(
; NOOP-LABEL: define {{.*}} @fms_kernel(
; FP64-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; NOPROFILE-NOT: call void @
; NOOP-NOT: call void @
; FP64: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; FP64-NEXT: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[A]], i64 %[[I]]
; FP64-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]]
; FP64-NEXT: %[[YP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[B]], i64 %[[I]]
; FP64-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]]
; FP64-NEXT: %[[T0:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[X]], double %[[Y]], double %[[X]])
; FP64-NEXT: %[[NY:[0-9]+]] = fneg double %[[Y]]
; FP64-NEXT: %[[T1:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T0]], double %[[T0]], double %[[NY]])
; FP64-NEXT: %[[T2:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T1]], double %[[Y]], double %[[T0]])
; FP64-NEXT: %[[NX:[0-9]+]] = fneg double %[[X]]
; FP64-NEXT: %[[T3:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T2]], double %[[T2]], double %[[NX]])
; FP64-NEXT: %[[T4:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T3]], double %[[T1]], double %[[T2]])
; FP64-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; FP64-NEXT: store double %[[T4]], ptr %[[OP]]
; FP64-NEXT: ret void
; NOPROFILE-NOT: _poseidon_body
; NOOP-NOT: _poseidon_body

; INLINED-LABEL: define {{.*}} @fms_kernel(
; INLINED-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; INLINED-NOT: call void @
; INLINED: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; INLINED: fptrunc double %{{[0-9]+}} to float
; INLINED-NOT: fmul double
; INLINED: [[J:%[^ ]+]] = fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join !{{[0-9]+}}
; INLINED-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; INLINED-NEXT: store double [[J]], ptr %[[OP]]
; INLINED-NEXT: ret void
; INLINED-NOT: _poseidon_body

; Three such kernels with one body and profile, differing only in the
; annotation: a target of 1e-7 takes FP32, 1e-12 the two-component expansion,
; and a bare annotation with no profile is left as written.
; TAU-ERR: [poseidon] preprocess_fms_loose_poseidon_body: tau=1.000000e-07 {{.*}} rel=1.428768e-08
; TAU-ERR: Applying solution for precision change All FP64(0%) + FP32(100%)
; TAU-ERR: [poseidon] pin: fms_loose:P0=All FP64(0%) + FP32(100%)
; TAU-ERR: [poseidon] preprocess_fms_tight_poseidon_body: tau=1.000000e-12 {{.*}} rel=7.011882e-16
; TAU-ERR: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; TAU-ERR: [poseidon] pin: fms_tight:P0=All FP64(0%) + Expansion2(100%)
; TAU-ERR: warning: {{.*}}in function fms_bare {{.*}}Poseidon: no profile at {{.*}}/fms_bare.fpprofile:
; TAU-ERR-NOT: Applying solution

; The loose site runs the body in FP32 between one narrowing of each input and
; one widening of the result.
; TAU-IR-LABEL: define {{.*}} @fms_loose(
; TAU-IR-COUNT-2: fptrunc double %{{[0-9]+}} to float
; TAU-IR-NOT: double
; TAU-IR-COUNT-5: call float @llvm.fmuladd.f32(
; TAU-IR-NOT: double
; TAU-IR: [[R:%[0-9]+]] = fpext float %{{[0-9]+}} to double
; TAU-IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds
; TAU-IR-NEXT: store double [[R]], ptr [[OP]]
; TAU-IR-NEXT: ret void

; The tight site takes the two-component expansion and stores the sum of the last pair.
; TAU-IR-LABEL: define {{.*}} @fms_tight(
; TAU-IR: fptrunc double %{{[0-9]+}} to float
; TAU-IR-NOT: fmuladd.f64
; TAU-IR: [[J:%[^ ]+]] = fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join !{{[0-9]+}}
; TAU-IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds
; TAU-IR-NEXT: store double [[J]], ptr [[OP]]
; TAU-IR-NEXT: ret void

; The unprofiled site keeps its FP64 arithmetic and no outlined body.
; TAU-IR-LABEL: define {{.*}} @fms_bare(
; TAU-IR-SAME: ptr {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[B:%[0-9]+]], ptr {{.*}}[[OUT:%[0-9]+]])
; TAU-IR-NOT: call void @
; TAU-IR: [[I:%[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; TAU-IR-NEXT: [[AP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[A]], i64 [[I]]
; TAU-IR-NEXT: [[AV:%[0-9]+]] = load double, ptr [[AP]]
; TAU-IR-NEXT: [[BP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[B]], i64 [[I]]
; TAU-IR-NEXT: [[BV:%[0-9]+]] = load double, ptr [[BP]]
; TAU-IR-NEXT: [[T0:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[AV]], double [[BV]], double [[AV]])
; TAU-IR-NEXT: [[NB:%[0-9]+]] = fneg double [[BV]]
; TAU-IR-NEXT: [[T1:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T0]], double [[T0]], double [[NB]])
; TAU-IR-NEXT: [[T2:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T1]], double [[BV]], double [[T0]])
; TAU-IR-NEXT: [[NA:%[0-9]+]] = fneg double [[AV]]
; TAU-IR-NEXT: [[T3:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T2]], double [[T2]], double [[NA]])
; TAU-IR-NEXT: [[T4:%[0-9]+]] = call double @llvm.fmuladd.f64(double [[T3]], double [[T1]], double [[T2]])
; TAU-IR-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[OUT]], i64 [[I]]
; TAU-IR-NEXT: store double [[T4]], ptr [[OP]]
; TAU-IR-NEXT: ret void
; TAU-IR-NOT: _poseidon_body

;--- kernel.ll
source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [20 x i8] c"annotated_kernel.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

;--- tau.ll
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

;--- body.ll
define dso_local ptx_kernel void @KERNEL(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
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

;--- decls.ll
declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
