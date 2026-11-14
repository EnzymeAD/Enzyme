; RUN: rm -rf %t && mkdir -p %t/prune %t/exact
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/cachedHerbieOutput_default_06F0ECDBFA1A458D.txt %t/prune/cachedHerbieOutput_cuda-sm120_8F24F8744AC078E0.txt
; RUN: cp %t/prune/cachedHerbieOutput_cuda-sm120_8F24F8744AC078E0.txt %t/exact/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/prune -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRUNE %s
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/exact -poseidon-early-prune=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=EXACT %s
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt
; The mode is part of the table's key: neither mode replays a table the other
; solved, and each replays its own.
; RUN: cp -r %t/prune %t/reuse
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/reuse -poseidon-early-prune=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=TO-EXACT %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/reuse -S -o /dev/null 2>&1 | FileCheck --check-prefix=TO-PRUNE %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/exact -poseidon-early-prune=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=REPLAY %s

; The binary64 and binary32 Herbie searches of one output are two units, and
; only one rewrite of the output can be taken. The binary64 rewrite (an FMA, a
; multiply, a min and a max) is cheaper than the original (an add and a
; multiply) on a cost model whose FP64 FMA costs 7/8 of an FP64 add, and more
; accurate, so pruning after its unit drops the empty strategy, and the binary32
; rewrite, much cheaper and less accurate, is refused by every strategy left.
; The exact solve keeps the empty strategy for the second unit and finds the
; cheaper point, the only one within the budget.

source_filename = "exact_dp_one_output.cu"
target triple = "nvptx64-nvidia-cuda"

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="sm_120" }

; PRUNE: Using cached Herbie output
; PRUNE: DP table contains 1 entries.
; PRUNE: No solution found within the computation cost budget!
; PRUNE-NOT: Applying solution
; PRUNE-BUDGETS: {{^}}-0.05859375{{$}}

; EXACT: Exact DP: 2 frontier points
; EXACT: DP table contains 2 entries.
; EXACT: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0))
; EXACT-BUDGETS: {{^}}-0.748046875,-0.05859375{{$}}

; TO-EXACT: DP cache entry for preprocess_tester was solved for a different -poseidon-early-prune; re-solving.
; TO-EXACT: DP table contains 2 entries.
; TO-PRUNE: DP cache entry for preprocess_tester was solved for a different -poseidon-early-prune; re-solving.
; TO-PRUNE: DP table contains 1 entries.
; REPLAY: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; REPLAY: DP table contains 2 entries.
