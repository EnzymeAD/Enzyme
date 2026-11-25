; DEFINE: %{fma} = -poseidon-enable-rewrite=true -poseidon-herbie-binary=/nonexistent/herbie \
; DEFINE:   -poseidon-enable-precision=false -S
; RUN: rm -rf %t && mkdir %t

; The exact solve (-poseidon-early-prune=false) against the frontier pruned
; after each unit, on the modules in Inputs/no_early_prune. Each pruned solve here
; drops a point the exact one keeps.

; one_output.ll: the binary64 and binary32 Herbie searches of one output are two
; units, and only one rewrite of the output can be taken. The binary64 rewrite
; (an FMA, a multiply, a min and a max) is cheaper than the original (an add and
; a multiply) on a cost model whose FP64 FMA costs 7/8 of an FP64 add, and more
; accurate, so pruning after its unit drops the empty strategy, and the binary32
; rewrite, much cheaper and less accurate, is refused by every strategy left.
; The exact solve keeps the empty strategy for the second unit and finds the
; cheaper point, the only one within the budget.
; RUN: mkdir %t/one_output %t/one_output/prune %t/one_output/exact
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/one_output/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/one_output/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/herbie_default_958ABA9C1BC0AE00.json %t/one_output/prune/herbie_cuda-sm120_45B4DF20BF2EFFF8.json
; RUN: cp %t/one_output/prune/herbie_cuda-sm120_45B4DF20BF2EFFF8.json %t/one_output/exact/
; RUN: %opt < %S/../Inputs/no_early_prune/one_output.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile %{fma} -poseidon-cost-model=%t/one_output/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/one_output/prune -poseidon-explain=%t/one_output/prune/ex -o /dev/null 2>&1 | FileCheck --check-prefix=ONE-PRUNE %s
; RUN: %poseidon_explain %t/one_output/prune/ex --format budgets | FileCheck --check-prefix=ONE-PRUNE-BUDGETS %s
; RUN: %opt < %S/../Inputs/no_early_prune/one_output.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile %{fma} -poseidon-cost-model=%t/one_output/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/one_output/exact -poseidon-explain=%t/one_output/exact/ex -poseidon-early-prune=false -o /dev/null 2>&1 | FileCheck --check-prefix=ONE-EXACT %s
; RUN: %poseidon_explain %t/one_output/exact/ex --format budgets | FileCheck --check-prefix=ONE-EXACT-BUDGETS %s
; The mode is part of the table's key: each mode solves into an entry of its
; own, and each replays its own.
; RUN: cp -r %t/one_output/prune %t/one_output/reuse
; RUN: %opt < %S/../Inputs/no_early_prune/one_output.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile %{fma} -poseidon-cost-model=%t/one_output/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/one_output/reuse -poseidon-early-prune=false -o /dev/null 2>&1 | FileCheck --check-prefix=ONE-TO-EXACT %s
; RUN: %opt < %S/../Inputs/no_early_prune/one_output.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile %{fma} -poseidon-cost-model=%t/one_output/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/one_output/reuse -o /dev/null 2>&1 | FileCheck --check-prefix=ONE-TO-PRUNE %s
; RUN: %opt < %S/../Inputs/no_early_prune/one_output.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile %{fma} -poseidon-cost-model=%t/one_output/cm_sm_120_fma.csv -poseidon-cost-budget=-0.7 -poseidon-cache=%t/one_output/exact -poseidon-early-prune=false -o /dev/null 2>&1 | FileCheck --check-prefix=ONE-REPLAY %s

; ONE-PRUNE: [poseidon] Using cached Herbie output
; ONE-PRUNE: DP table contains 1 entries.
; ONE-PRUNE: [poseidon] No solution found within the computation cost budget
; ONE-PRUNE-NOT: Applying solution
; ONE-PRUNE: [poseidon] no rewrite applied for preprocess_tester; calling original body tester (bit-transparent no-op)
; ONE-PRUNE-BUDGETS: {{^}}-0.05859375{{$}}
; ONE-EXACT: [poseidon] Exact DP: 2 frontier points; peak 2 entries in 2 signatures; 0 precision changes priced; {{.*}} s
; ONE-EXACT: DP table contains 2 entries.
; ONE-EXACT: Minimum accuracy cost within budget: 7.671461e-05
; ONE-EXACT-NEXT: Computation cost budget used: -7.480469e-01
; ONE-EXACT: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0))
; ONE-EXACT: [poseidon] pin: tester:R1=(fma.f32 v1 v2 (*.f32 v2 v0))
; ONE-EXACT-BUDGETS: {{^}}-0.748046875,-0.05859375{{$}}
; ONE-TO-EXACT-NOT: DP cache entry
; ONE-TO-EXACT: Solving the DP for preprocess_tester.
; ONE-TO-EXACT: DP table contains 2 entries.
; ONE-TO-PRUNE-NOT: DP cache entry
; ONE-TO-PRUNE: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; ONE-TO-PRUNE: DP table contains 1 entries.
; ONE-REPLAY: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; ONE-REPLAY: DP table contains 2 entries.

; joint.ll, two sites of one_output.ll's kind, on its cost model (the synthetic
; one with the FP64 FMA lowered to 7/8 of an FP64 add, so that the binary64
; rewrite is cheaper than the original), under one shared budget. Pruning after
; each site's binary64 rewrite drops every strategy without it, so neither
; site's binary32 rewrite reaches the joint frontier. The exact frontier is the
; sum of the two sites' own: both binary32 rewrites, one of each kind, or both
; binary64 ones.
; RUN: mkdir %t/joint %t/joint/prune %t/joint/exact %t/joint/prof
; RUN: cp %S/../Inputs/fma_site_profile/tester.fpprofile %t/joint/prof/tester_a.fpprofile
; RUN: cp %S/../Inputs/fma_site_profile/tester.fpprofile %t/joint/prof/tester_b.fpprofile
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/joint/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/joint/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/herbie_default_958ABA9C1BC0AE00.json %t/joint/prune/herbie_cuda-sm120_45B4DF20BF2EFFF8.json
; RUN: cp %t/joint/prune/herbie_cuda-sm120_45B4DF20BF2EFFF8.json %t/joint/exact/
; RUN: %opt < %S/../Inputs/no_early_prune/joint.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%t/joint/prof %{fma} -poseidon-cost-model=%t/joint/cm_sm_120_fma.csv -poseidon-solve-jointly -poseidon-cost-budget=-0.7 -poseidon-cache=%t/joint/prune -poseidon-explain=%t/joint/prune/ex -o /dev/null 2>&1 | FileCheck --check-prefix=JOINT-PRUNE %s
; RUN: %poseidon_explain %t/joint/prune/ex --format budgets | FileCheck --check-prefix=JOINT-PRUNE-BUDGETS %s
; RUN: %opt < %S/../Inputs/no_early_prune/joint.ll %loadPoseidon -poseidon-print -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/joint/prof %{fma} -poseidon-cost-model=%t/joint/cm_sm_120_fma.csv -poseidon-solve-jointly -poseidon-cost-budget=-0.7 -poseidon-cache=%t/joint/exact -poseidon-explain=%t/joint/exact/ex -poseidon-early-prune=false -o /dev/null 2>&1 | FileCheck --check-prefix=JOINT-EXACT %s
; RUN: %poseidon_explain %t/joint/exact/ex --format budgets | FileCheck --check-prefix=JOINT-EXACT-BUDGETS %s

; JOINT-PRUNE: DP table contains 1 entries.
; JOINT-PRUNE: [poseidon] No solution found within the computation cost budget
; JOINT-PRUNE: [poseidon] no rewrite applied for preprocess_tester_a; wrapper call restored to original body tester_a (bit-transparent no-op)
; JOINT-PRUNE: [poseidon] no rewrite applied for preprocess_tester_b; wrapper call restored to original body tester_b (bit-transparent no-op)
; JOINT-PRUNE-BUDGETS: {{^}}-0.05859375{{$}}
; JOINT-EXACT: [poseidon] Exact DP: 3 frontier points; peak 4 entries in 2 signatures; 0 precision changes priced; {{.*}} s
; JOINT-EXACT: Critical computation cost range: [-7.480469e-01, -5.859375e-02]
; JOINT-EXACT-NEXT: DP table contains 3 entries.
; JOINT-EXACT: Minimum accuracy cost within budget: 1.534292e-04
; JOINT-EXACT-NEXT: Computation cost budget used: -7.480469e-01
; JOINT-EXACT-COUNT-2: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0))
; JOINT-EXACT-BUDGETS: {{^}}-0.748046875,-0.4033203125,-0.05859375{{$}}

; refusal.ll, footprint refusal. The FP subgraph of the reduction loop's FMA
; also holds the epilogue D = acc * ((alpha + 1) - alpha), whose Herbie rewrite
; acc is cheaper and more accurate than the original. Pruning after the CO unit
; drops the empty strategy, the rewrite refuses the overlapping raise, and no
; raise reaches the frontier; the exact solve finds all three. A budget below
; every frontier point selects nothing, so only the solve runs.
; RUN: mkdir %t/refusal
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/refusal_herbie %opt %S/../Inputs/no_early_prune/refusal.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/refusal -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/refusal/prune -poseidon-explain=%t/refusal/prune/ex 2>&1 | FileCheck --check-prefix=REFUSAL-PRUNE %s
; RUN: %poseidon_explain %t/refusal/prune/ex --format budgets | FileCheck --check-prefix=REFUSAL-PRUNE-BUDGETS %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/refusal_herbie %opt %S/../Inputs/no_early_prune/refusal.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/refusal -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/refusal/exact -poseidon-explain=%t/refusal/exact/ex -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=REFUSAL-EXACT %s
; RUN: %poseidon_explain %t/refusal/exact/ex --format budgets | FileCheck --check-prefix=REFUSAL-EXACT-BUDGETS %s

; REFUSAL-PRUNE: [poseidon] matmul[0] candidates
; REFUSAL-PRUNE-NEXT: wmma m16n16k16 f16/f32: dcost=-3.932160e+06
; REFUSAL-PRUNE-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06
; REFUSAL-PRUNE-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06
; REFUSAL-PRUNE: DP table contains 1 entries.
; REFUSAL-PRUNE-BUDGETS: {{^}}-0.10526315789473684{{$}}
; REFUSAL-EXACT: [poseidon] Exact DP: 4 frontier points
; REFUSAL-EXACT: DP table contains 4 entries.
; REFUSAL-EXACT-BUDGETS: {{^}}-0.78947368421052633,-0.77631578947368418,-0.76315789473684215,-0.10526315789473684{{$}}

; reprice.ll, a precision change priced against the rewrites taken. The kernel
; computes the matrix product with its epilogue D = acc * alpha, then Y, an FP
; subgraph the raise m does not touch. Herbie's cheaper answer x for Y
; (Inputs/no_early_prune/reprice_herbie) is cheaper and more accurate than m alone, so
; pruning after the CM unit drops every strategy that raises m without rewriting
; Y. Y has one output, so its precision changes u change nothing once Y is
; rewritten, and u never meets m: the four cheapest points of the exact
; frontier, {u, m}, are missing. IR and profile are written by hand, the math
; functions as the libdevice calls the pass recognizes. A budget below every
; frontier point selects nothing, so only the solve runs.
; RUN: mkdir %t/reprice
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/reprice/prune -poseidon-explain=%t/reprice/prune/ex 2>&1 | FileCheck --check-prefix=REPRICE-PRUNE %s
; RUN: %poseidon_explain %t/reprice/prune/ex --format budgets | FileCheck --check-prefix=REPRICE-PRUNE-BUDGETS %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/reprice/exact -poseidon-explain=%t/reprice/exact/ex -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=REPRICE-EXACT %s
; RUN: %poseidon_explain %t/reprice/exact/ex --format budgets | FileCheck --check-prefix=REPRICE-EXACT-BUDGETS %s

; REPRICE-PRUNE: DP table contains 17 entries.
; REPRICE-PRUNE-BUDGETS-NOT: -0.95376420454545452
; REPRICE-PRUNE-BUDGETS: {{^}}-0.70823863636363638,-0.69236505681818183,-0.68103693181818181,-0.49815340909090911,-0.48227982954545456,-0.47095170454545454,-0.40106534090909091,-0.36573153409090908,-0.34985795454545454,-0.33852982954545452,-0.27727272727272728,-0.20738636363636365,-0.13750000000000001,-0.067613636363636362,0,0.13412642045454545,0.14545454545454545{{$}}
; REPRICE-EXACT: [poseidon] Exact DP: 21 frontier points
; REPRICE-EXACT: DP table contains 21 entries.
; REPRICE-EXACT-BUDGETS: {{^}}-0.95376420454545452,-0.94921875,-0.94467329545454548,-0.73458806818181821,-0.70823863636363638,-0.69236505681818183,-0.68103693181818181,-0.49815340909090911,-0.48227982954545456,-0.47095170454545454,-0.40106534090909091,-0.36573153409090908,-0.34985795454545454,-0.33852982954545452,-0.27727272727272728,-0.20738636363636365,-0.13750000000000001,-0.067613636363636362,0,0.13412642045454545,0.14545454545454545{{$}}

; two_matmuls.ll, two matrix products with one footprint, d ten times less
; sensitive than e. The native DGEMM dispatch of e has zero modelled error, so
; pruning after e keeps no strategy without it and refuses d; the exact solve
; (-poseidon-early-prune=false) keeps d's options. The budget is below every
; frontier point, so only the solve runs.
; RUN: mkdir %t/two_matmuls
; RUN: %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/two_matmuls/prune 2>&1 | FileCheck --check-prefix=TWO-PRUNE %s
; RUN: cat %t/two_matmuls/prune/dp_*.json | FileCheck --check-prefix=TWO-PRUNE-TABLE %s
; RUN: %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cost-budget=-1 -poseidon-print -S -o /dev/null -poseidon-cache=%t/two_matmuls/exact -poseidon-early-prune=false 2>&1 | FileCheck --check-prefix=TWO-EXACT %s
; RUN: cat %t/two_matmuls/exact/dp_*.json | FileCheck --check-prefix=TWO-EXACT-TABLE %s

; TWO-PRUNE: DP table contains 9 entries.
; TWO-PRUNE-TABLE: "accuracy_cost": 1633.8211240338183{{[0-9]*}},
; TWO-PRUNE-TABLE-NEXT: "cost": -4261412864,
; TWO-PRUNE-TABLE-NOT: "index": 1
; TWO-EXACT: [poseidon] Exact DP: 9 frontier points
; TWO-EXACT: DP table contains 9 entries.
; TWO-EXACT-TABLE: "accuracy_cost": 163.38211240338188{{[0-9]*}},
; TWO-EXACT-TABLE-NEXT: "cost": -4261412864,
; TWO-EXACT-TABLE: "index": 1
