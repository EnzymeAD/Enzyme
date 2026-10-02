; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s
; REQUIRES: poseidon, enzyme

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y, double %z) {
entry:
  %0 = fmul fast double %x, %y
  %1 = fadd fast double %0, %z
  ret double %1
}

define double @test_profile(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z)
  ret double %0
}

; Function Attrs: nounwind
declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

; CHECK: @POSEIDON_PROFILE_RUNTIME_VAR = external global i32
; CHECK: @poseidon_site_preprocess_tester = private unnamed_addr constant [18 x i8] c"preprocess_tester\00", align 1

; CHECK: define internal double @__poseidon_probe_preprocess_tester_0_rev(double %[[v:.+]], double %[[dret:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_tester, i64 0, double %[[v]], double %[[dret]])
; CHECK-NEXT:   ret double %[[dret]]

; CHECK: define internal { double, double, double } @diffepreprocess_tester(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [3 x double], align 8
; CHECK-NEXT:   %[[fmuladd:.+]] = call fast double @llvm.fmuladd.f64(double %x, double %y, double %z){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0:[0-9]+]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_tester_0_aug(double %[[fmuladd]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr [3 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr [3 x double], ptr %[[alloca]], i32 0, i32 2
; CHECK-NEXT:   store double %z, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_tester, i64 0, double %[[probed]], i32 3, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_tester_0(double %[[fmuladd]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[grad]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[grad]], %x
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double, double } undef, double %[[grad1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double, double } %[[ins1]], double %[[grad2]], 1
; CHECK-NEXT:   %[[ins3:.+]] = insertvalue { double, double, double } %[[ins2]], double %[[grad]], 2
; CHECK-NEXT:   ret { double, double, double } %[[ins3]]
; CHECK-NEXT: }

; CHECK: ![[idx0]] = !{i64 0}
