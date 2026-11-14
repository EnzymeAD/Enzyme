; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -S -poseidon-cache=%t/both -poseidon-comp-cost-budget=-400 2>&1 | FileCheck --check-prefix=BOTH %s
; RUN: FileCheck --check-prefix=BUDGETS %s < %t/both/budgets.txt
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -S -poseidon-cache=%t/one -poseidon-comp-cost-budget=-300 2>&1 | FileCheck --check-prefix=ONE %s
; REQUIRES: poseidon
; Two annotated sites solved under one shared budget (-poseidon-joint-dp): one budgets.txt lists the joint table's breakpoints, a budget of -400 rewrites both sites to FP32, a budget of -300 is spent on tester_b alone and site_a is restored to its original body. Profiles from running this module through the host profiler with Inputs/joint_profiles/joint_driver.c.

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
; BOTH: JOINT DP table contains 4 entries; cost range [-576, 0]
; BOTH: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -576
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

; BUDGETS: -576,-384,-64,0

; ONE: JOINT DP table contains 4 entries; cost range [-576, 0]
; ONE: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -384
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
