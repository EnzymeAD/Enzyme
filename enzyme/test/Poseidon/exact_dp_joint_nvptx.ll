; RUN: rm -rf %t && mkdir -p %t/prune %t/exact
; RUN: cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_default_06F0ECDBFA1A458D.txt %t/prune/cachedHerbieOutput_cuda-sm120_B8F90965B34ECC31.txt
; RUN: cp %t/prune/cachedHerbieOutput_cuda-sm120_B8F90965B34ECC31.txt %t/exact/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/joint -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-joint-dp -poseidon-comp-cost-budget=-10000000 -poseidon-cache=%t/prune -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/exact_dp/joint -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-joint-dp -poseidon-comp-cost-budget=-10000000 -poseidon-cache=%t/exact -poseidon-early-prune=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt

; Two sites of exact_dp_one_output_nvptx.ll's kind under one shared budget.
; Pruning after each site's binary64 rewrite drops every strategy without it,
; so neither site's binary32 rewrite reaches the joint frontier. The exact
; frontier is the sum of the two sites' own: both binary32 rewrites, one of
; each kind, or both binary64 ones.

source_filename = "exact_dp_joint.cu"
target triple = "nvptx64-nvidia-cuda"

define double @tester_a(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @tester_b(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @site_a(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester_a, double %x, double %y, double %z)
  ret double %0
}

define double @site_b(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester_b, double %x, double %y, double %z)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="sm_120" }

; PRUNE: JOINT DP table contains 1 entries
; PRUNE: JOINT: no solution within the computation cost budget!
; PRUNE-BUDGETS: {{^}}-94930{{$}}

; EXACT: Exact DP: 3 frontier points
; EXACT: JOINT DP table contains 3 entries
; EXACT: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -15787528
; EXACT-COUNT-2: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0))
; EXACT-BUDGETS: {{^}}-15787528,-7941229,-94930{{$}}
