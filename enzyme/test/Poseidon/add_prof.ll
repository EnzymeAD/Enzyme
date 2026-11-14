; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s -dump-input=always
; REQUIRES: poseidon, enzyme

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_profile(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y, metadata !"poseidon_tau", double 1.0e-6)
  ret double %0
}

; Function Attrs: nounwind
declare double @__poseidon_fp_optimize(double (double, double)*, ...)

; CHECK: @POSEIDON_PROFILE_RUNTIME_VAR = external global i32
; CHECK: @poseidon_site_preprocess_tester = private unnamed_addr constant [18 x i8] c"preprocess_tester\00", align 1

; The adjoint of slot 0 leaves the reverse pass through the probe's custom
; gradient, which is where the gradient record is written.
; CHECK: define internal double @__poseidon_probe_preprocess_tester_0_rev(double %[[v:.+]], double %[[dret:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_tester, i64 0, double %[[v]], double %[[dret]])
; CHECK-NEXT:   ret double %[[dret]]

; CHECK: define internal double @__poseidon_probe_preprocess_tester_0(double %{{.+}}) #{{[0-9]+}} !enzyme_augment !{{[0-9]+}} !enzyme_gradient !{{[0-9]+}} {

; CHECK: define internal { double, double } @diffepreprocess_tester(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[fadd:.+]] = fadd fast double %x, %y, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0:[0-9]+]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_tester_0_aug(double %[[fadd]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_tester, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_tester_0(double %[[fadd]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[grad]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[grad]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

; CHECK: ![[idx0]] = !{i64 0}
