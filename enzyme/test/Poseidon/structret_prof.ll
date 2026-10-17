; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s
; REQUIRES: poseidon, enzyme

; Adapted from enzyme/test/Enzyme/ReverseMode/gradient-struct-ret.ll

%struct.Gradients = type { double, double }

; Function Attrs: noinline nounwind readnone uwtable
define dso_local double @muldd(double %x, double %y) {
entry:
  %mul = fmul fast double %x, %y
  ret double %mul
}

define dso_local %struct.Gradients @test_profile(double %x, double %y) {
entry:
  %call = call %struct.Gradients (i8*, ...) @__poseidon_fp_optimize(i8* bitcast (double (double, double)* @muldd to i8*), double %x, double %y)
  ret %struct.Gradients %call
}

; Function Attrs: nounwind
declare %struct.Gradients @__poseidon_fp_optimize(i8*, ...)

; CHECK: @POSEIDON_PROFILE_RUNTIME_VAR = external global i32
; CHECK: @poseidon_site_preprocess_muldd = private unnamed_addr constant [17 x i8] c"preprocess_muldd\00", align 1

; CHECK: define internal double @__poseidon_probe_preprocess_muldd_0_rev(double %[[v:.+]], double %[[dret:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_muldd, i64 0, double %[[v]], double %[[dret]])
; CHECK-NEXT:   ret double %[[dret]]

; CHECK: define internal { double, double } @diffepreprocess_muldd(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %x, %y, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0:[0-9]+]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_muldd_0_aug(double %[[mul]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_muldd, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_muldd_0(double %[[mul]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[grad]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[grad]], %x
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[grad1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[grad2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

; CHECK: ![[idx0]] = !{i64 0}
