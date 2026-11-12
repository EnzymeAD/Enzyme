; RUN: rm -rf %t && mkdir -p %t/prune %t/exact
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/cachedHerbieOutput_default_06F0ECDBFA1A458D.txt %t/prune/cachedHerbieOutput_cuda-sm120_8F24F8744AC078E0.txt
; RUN: cp %t/prune/cachedHerbieOutput_cuda-sm120_8F24F8744AC078E0.txt %t/exact/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/exact_dp/joint -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-joint-dp -poseidon-cost-budget=-0.7 -poseidon-cache=%t/prune -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/exact_dp/joint -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-joint-dp -poseidon-cost-budget=-0.7 -poseidon-cache=%t/exact -poseidon-early-prune=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt

; Two sites of exact_dp_one_output_nvptx.ll's kind, on its cost model (the
; synthetic one with the FP64 FMA lowered to 7/8 of an FP64 add, so that the
; binary64 rewrite is cheaper than the original), under one shared budget.
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
; PRUNE-BUDGETS: {{^}}-0.05859375{{$}}

; EXACT: Exact DP: 3 frontier points
; EXACT: JOINT DP table contains 3 entries
; EXACT: JOINT: minimum accuracy cost within budget: {{.*}}; computation cost used: -7.480469e-01
; EXACT-COUNT-2: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0))
; EXACT-BUDGETS: {{^}}-0.748046875,-0.4033203125,-0.05859375{{$}}
