; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-generate -S | FileCheck %s
; REQUIRES: poseidon, enzyme

define double @test_maxnum_zero(double %x) {
entry:
  %cmp = fcmp ogt double %x, 0.0
  %result = select i1 %cmp, double %x, double 0.0
  ret double %result
}

define double @test_maxnum_zero_reversed(double %x) {
entry:
  %cmp = fcmp olt double %x, 0.0
  %result = select i1 %cmp, double 0.0, double %x
  ret double %result
}

define double @test_maxnum_general(double %x, double %y) {
entry:
  %cmp = fcmp ogt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

define double @test_minnum_general(double %x, double %y) {
entry:
  %cmp = fcmp olt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

define double @test_combined(double %x, double %y, double %z) {
entry:
  %cmp1 = fcmp ogt double %x, 0.0
  %max_x = select i1 %cmp1, double %x, double 0.0
  %cmp2 = fcmp olt double %max_x, %y
  %min_xy = select i1 %cmp2, double %max_x, double %y
  %result = fadd fast double %min_xy, %z
  ret double %result
}

define double @test_profile_maxnum_zero(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__poseidon_fp_optimize(double (double)* nonnull @test_maxnum_zero, double %x)
  ret double %0
}

define double @test_profile_maxnum_zero_reversed(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__poseidon_fp_optimize(double (double)* nonnull @test_maxnum_zero_reversed, double %x)
  ret double %0
}

define double @test_profile_maxnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @test_maxnum_general, double %x, double %y)
  ret double %0
}

define double @test_profile_minnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @test_minnum_general, double %x, double %y)
  ret double %0
}

define double @test_profile_combined(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @test_combined, double %x, double %y, double %z)
  ret double %0
}

; Function Attrs: nounwind
declare double @__poseidon_fp_optimize(...)

; CHECK: @POSEIDON_PROFILE_RUNTIME_VAR = external global i32
; CHECK: @poseidon_site_preprocess_test_maxnum_zero = private unnamed_addr constant [28 x i8] c"preprocess_test_maxnum_zero\00", align 1
; CHECK: @poseidon_site_preprocess_test_maxnum_zero_reversed = private unnamed_addr constant [37 x i8] c"preprocess_test_maxnum_zero_reversed\00", align 1
; CHECK: @poseidon_site_preprocess_test_maxnum_general = private unnamed_addr constant [31 x i8] c"preprocess_test_maxnum_general\00", align 1
; CHECK: @poseidon_site_preprocess_test_minnum_general = private unnamed_addr constant [31 x i8] c"preprocess_test_minnum_general\00", align 1
; CHECK: @poseidon_site_preprocess_test_combined = private unnamed_addr constant [25 x i8] c"preprocess_test_combined\00", align 1

; One probe per profiled value, and its custom gradient carries that value's
; slot into the gradient record.
; CHECK: define internal double @__poseidon_probe_preprocess_test_maxnum_zero_0_rev(double %[[v0:.+]], double %[[d0:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_maxnum_zero, i64 0, double %[[v0]], double %[[d0]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_maxnum_zero_reversed_0_rev(double %[[v1:.+]], double %[[d1:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_maxnum_zero_reversed, i64 0, double %[[v1]], double %[[d1]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_maxnum_general_0_rev(double %[[v2:.+]], double %[[d2:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_maxnum_general, i64 0, double %[[v2]], double %[[d2]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_minnum_general_0_rev(double %[[v3:.+]], double %[[d3:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_minnum_general, i64 0, double %[[v3]], double %[[d3]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_0_rev(double %[[v4:.+]], double %[[d4:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_combined, i64 0, double %[[v4]], double %[[d4]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_1_rev(double %[[v5:.+]], double %[[d5:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_combined, i64 1, double %[[v5]], double %[[d5]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_2_rev(double %[[v6:.+]], double %[[d6:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @poseidonLogGrad(ptr @poseidon_site_preprocess_test_combined, i64 2, double %[[v6]], double %[[d6]])

; CHECK: define internal { double } @diffepreprocess_test_maxnum_zero(double %x, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0:[0-9]+]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_maxnum_zero_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_maxnum_zero, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_maxnum_zero_0(double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[sel:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[grad]]
; CHECK-NEXT:   %[[ins:.+]] = insertvalue { double } undef, double %[[sel]], 0
; CHECK-NEXT:   ret { double } %[[ins]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffepreprocess_test_maxnum_zero_reversed(double %x, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_maxnum_zero_reversed_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_maxnum_zero_reversed, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_maxnum_zero_reversed_0(double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[sel:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[grad]]
; CHECK-NEXT:   %[[ins:.+]] = insertvalue { double } undef, double %[[sel]], 0
; CHECK-NEXT:   ret { double } %[[ins]]
; CHECK-NEXT: }

; CHECK: define internal { double, double } @diffepreprocess_test_maxnum_general(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_maxnum_general_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_maxnum_general, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_maxnum_general_0(double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel1:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[grad]]
; CHECK-NEXT:   %[[cmp2:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel2:.+]] = select fast i1 %[[cmp2]], double %[[grad]], double 0.000000e+00
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[sel1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[sel2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

; CHECK: define internal { double, double } @diffepreprocess_test_minnum_general(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[minnum:.+]] = call double @llvm.minnum.f64(double %x, double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_minnum_general_0_aug(double %[[minnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_minnum_general, i64 0, double %[[probed]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_minnum_general_0(double %[[minnum]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel1:.+]] = select fast i1 %[[cmp]], double %[[grad]], double 0.000000e+00
; CHECK-NEXT:   %[[cmp2:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel2:.+]] = select fast i1 %[[cmp2]], double 0.000000e+00, double %[[grad]]
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[sel1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[sel2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

; Three profiled values in one site: each is logged under its own slot and each
; operand array records the probed (not the raw) value of its producer.
; CHECK: define internal { double, double, double } @diffepreprocess_test_combined(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca0:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca1:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca2:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx0]]
; CHECK-NEXT:   %[[probed0:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca0]], align 8
; CHECK-NEXT:   %[[gep0:.+]] = getelementptr [2 x double], ptr %[[alloca0]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep0]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_combined, i64 0, double %[[probed0]], i32 2, ptr %[[alloca0]])
; CHECK-NEXT:   %[[minnum:.+]] = call double @llvm.minnum.f64(double %[[probed0]], double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx1:[0-9]+]]
; CHECK-NEXT:   %[[probed1:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_1_aug(double %[[minnum]])
; CHECK-NEXT:   store double %[[probed0]], ptr %[[alloca1]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr [2 x double], ptr %[[alloca1]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_combined, i64 1, double %[[probed1]], i32 2, ptr %[[alloca1]])
; CHECK-NEXT:   %result = fadd fast double %[[probed1]], %z, !enzyme_active !{{[0-9]+}}, !poseidon.prof.idx ![[idx2:[0-9]+]]
; CHECK-NEXT:   %[[probed2:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_2_aug(double %result)
; CHECK-NEXT:   store double %[[probed1]], ptr %[[alloca2]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr [2 x double], ptr %[[alloca2]], i32 0, i32 1
; CHECK-NEXT:   store double %z, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @poseidonLogValue(ptr @poseidon_site_preprocess_test_combined, i64 2, double %[[probed2]], i32 2, ptr %[[alloca2]])
; CHECK-NEXT:   %[[gc2:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_combined_2(double %result, double %[[differet]])
; CHECK-NEXT:   %[[g2:.+]] = extractvalue { double } %[[gc2]], 0
; CHECK-NEXT:   %[[gc1:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_combined_1(double %[[minnum]], double %[[g2]])
; CHECK-NEXT:   %[[g1:.+]] = extractvalue { double } %[[gc1]], 0

; CHECK: ![[idx0]] = !{i64 0}
; CHECK: ![[idx1]] = !{i64 1}
; CHECK: ![[idx2]] = !{i64 2}
