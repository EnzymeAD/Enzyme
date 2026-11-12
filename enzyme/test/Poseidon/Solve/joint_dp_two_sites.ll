; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%cost_models/cm_cpu_native.csv -S -poseidon-cache=%t/both -poseidon-cost-budget=-0.4 2>&1 | FileCheck --check-prefix=BOTH %s
; RUN: FileCheck --check-prefix=BUDGETS %s < %t/both/budgets.txt
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%cost_models/cm_cpu_native.csv -S -poseidon-cache=%t/one -poseidon-cost-budget=-0.2 2>&1 | FileCheck --check-prefix=ONE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=STALE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-joint-dp -poseidon-cost-budget=inf -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONFINITE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-cost-budget=1e300 -S -o /dev/null 2>&1 | FileCheck --check-prefix=HUGE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-cost-budget= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-joint-dp -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-tau= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-tau %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-min-cost-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-min-cost-share %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-max-broken-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-max-broken-share %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-exponent-penalty= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY -DFLAG=poseidon-exponent-penalty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -poseidon-tau= -poseidon-cost-budget= -poseidon-exponent-penalty= -S -o /dev/null 2>&1 | FileCheck --check-prefix=UNUSED --allow-empty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=UNUSED --allow-empty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -poseidon-tau=0.5 -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=UNUSED --allow-empty %s
; Two annotated sites solved under one shared budget (-poseidon-joint-dp): one budgets.txt lists the joint table's breakpoints, a budget of -0.4 rewrites both sites to FP32, a budget of -0.2 is spent on tester_b alone and site_a is restored to its original body. Profiles from running this module through the host profiler with Inputs/joint_profiles/joint_driver.c.

define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @tester_b(double %x, double %y) #0 {
entry:
  %mul = fmul fast double %x, %y
  %sub = fsub fast double %mul, %y
  %div = fdiv fast double %sub, %x
  ret double %div
}

define double @site_a(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y)
  ret double %0
}

define double @site_b(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_b, double %x, double %y)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; BOTH: Poseidon JOINT: collecting candidates from 2 marked function(s)
; BOTH: JOINT DP table contains 4 entries; cost range [-4.500000e-01, 0.000000e+00]
; BOTH: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -4.500000e-01
; BOTH: Applying solution for CS: All FP64(0%) + FP32(100%)
; BOTH: [poseidon] Finished optimizing preprocess_tester_a
; BOTH: Applying solution for CS: All FP64(0%) + FP32(100%)
; BOTH: [poseidon] Finished optimizing preprocess_tester_b
; BOTH-LABEL: define double @site_a(
; BOTH-NEXT: entry:
; BOTH-NEXT: call double @preprocess_tester_a(double %x, double %y)
; BOTH-LABEL: define double @site_b(
; BOTH-NEXT: entry:
; BOTH-NEXT: call double @preprocess_tester_b(double %x, double %y)
; BOTH-LABEL: define double @preprocess_tester_a(
; BOTH: fptrunc fast double %x to float
; BOTH: fadd fast float
; BOTH: fmul fast float
; BOTH: fpext float
; BOTH-LABEL: define double @preprocess_tester_b(
; BOTH: fptrunc fast double %y to float
; BOTH: fneg fast float
; BOTH: call fast float @llvm.fmuladd.f32(
; BOTH: fdiv fast float
; BOTH: fpext float

; BUDGETS: -0.45000000000000001,-0.29999999999999999,-0.050000000000000003,0

; ONE: JOINT DP table contains 4 entries; cost range [-4.500000e-01, 0.000000e+00]
; ONE: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -3.000000e-01
; ONE: [poseidon] Finished optimizing preprocess_tester_a
; ONE: [poseidon] no rewrite applied for preprocess_tester_a; wrapper call restored to original body tester_a
; ONE: Applying solution for CS: All FP64(0%) + FP32(100%)
; ONE: [poseidon] Finished optimizing preprocess_tester_b
; ONE-LABEL: define double @site_a(
; ONE-NEXT: entry:
; ONE-NEXT: call double @tester_a(double %x, double %y)
; ONE-LABEL: define double @site_b(
; ONE-NEXT: entry:
; ONE-NEXT: call double @preprocess_tester_b(double %x, double %y)
; ONE-NOT: define double @preprocess_tester_a(

; A cost given where the budget takes a share (-400) asks to save more than the
; whole profiled cost, which no solution can: a solve that selects by the budget
; stops with an error at the site rather than selecting nothing, as it does
; for a share that is not finite. A compile that solves nothing, or solves
; under an accuracy target, does not read the budget.
; STALE-NOT: Starting accuracy DP solver
; STALE: error: {{.*}}Poseidon: -poseidon-cost-budget=-400 asks a solution to save more than the whole profiled cost. It takes the share of a site's profiled cost a solution may add, negative for one it has to save (-0.25 requires saving a quarter), not a cost. A solve writes to budgets.txt in its -poseidon-cache directory the share of each frontier point a budget selects.
; HUGE: error: {{.*}}Poseidon: -poseidon-cost-budget=1.0000000000000001e+300 lets a solution add more than 2^30 times the profiled cost, which the DP's cost axis does not hold; give a share of at most 1073741824.
; NONFINITE-NOT: Starting accuracy DP solver
; NONFINITE: error: {{.*}}Poseidon: -poseidon-cost-budget=inf is not a finite number. It takes the share of a site's profiled cost
; UNUSED-NOT: error:
; UNUSED-NOT: -poseidon-cost-budget

; A number flag given no number, an empty value or nan, is an error at the
; first site whose solve reads it, not 0 (what LLVM's parser of a double makes
; of an empty value): the budget is read by a solve that selects by it, the
; other number flags by every solve.
; EMPTY-NOT: Starting accuracy DP solver
; EMPTY: error: {{.*}}Poseidon: -[[FLAG]] was given no number: its value is empty or nan. Write a number after '=', or leave the flag out for its default.
; EMPTY-NOT: Starting accuracy DP solver
