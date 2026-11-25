; RUN: rm -rf %t && mkdir -p %t/joint %t/jprof
; RUN: cp %S/../Inputs/fma_opt_profiles/preprocess_tester.fpprofile %t/jprof/preprocess_tester_a.fpprofile
; RUN: cp %S/../Inputs/fma_opt_profiles/preprocess_tester.fpprofile %t/jprof/preprocess_tester_b.fpprofile
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/regime_split %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-arm-search-factor=1024 -poseidon-cache=%t/noop -S -o /dev/null 2>&1 | FileCheck --check-prefix=NOOP --implicit-check-not=NOTE: %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/one_rewrite_per_output %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/step -S -o /dev/null 2>&1 | FileCheck --check-prefix=STEP --implicit-check-not=NOTE: %s
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/joint/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/joint/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/cachedHerbieOutput_default_958ABA9C1BC0AE00.txt %t/joint/cachedHerbieOutput_cuda-sm120_45B4DF20BF2EFFF8.txt
; RUN: %opt < %S/../Inputs/exact_dp/joint.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%t/jprof -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/joint/cm_sm_120_fma.csv -poseidon-joint-dp -poseidon-cache=%t/joint -S -o /dev/null 2>&1 | FileCheck --check-prefix=JOINT --implicit-check-not=NOTE: %s

; A DP frontier of one point is either the code as written, which no
; candidate improves on, or a solution that is no costlier and no less
; accurate than the code as written: one rewrite of the same cost and
; better accuracy (STEP), or two that are cheaper and more accurate in a
; joint solve (JOINT). The note names which.

; NOOP: Critical computation cost range: [0.000000e+00, 0.000000e+00]{{$}}
; NOOP-NEXT: DP table contains 1 entries.{{$}}
; NOOP-NEXT: [poseidon] NOTE: none of the 1 candidate(s), alone or combined, is cheaper or more accurate than the code as written, which is the only frontier point.{{$}}
; NOOP-NEXT: Total candidate compositions: 2.000000e+00
; NOOP-NEXT: WARNING: No-op solution (utilized cost budget = 0) is considered Pareto-optimal.{{$}}
; NOOP-NOT: Applying solution

; STEP: Critical computation cost range: [[[C:[-+.e0-9]+]], [[C]]]{{$}}
; STEP-NEXT: DP table contains 1 entries.{{$}}
; STEP-NEXT: [poseidon] NOTE: the only frontier point is a solution of 1 step(s) from the 2 candidate(s); it is no costlier and no less accurate than the code as written and every other combination.{{$}}
; STEP-NOT: No-op solution
; STEP: Computation cost budget used: [[C]]{{$}}
; STEP: !!! Applying solution (1 step(s)) ... !!!{{$}}

; JOINT: Critical computation cost range: [[[C:-[.e0-9-]+]], [[C]]]{{$}}
; JOINT-NEXT: DP table contains 1 entries.{{$}}
; JOINT-NEXT: [poseidon] NOTE: the only frontier point is a solution of 2 step(s) from the 4 candidate(s); it is no costlier and no less accurate than the code as written and every other combination.{{$}}
; JOINT: Computation cost budget used: [[C]]{{$}}
