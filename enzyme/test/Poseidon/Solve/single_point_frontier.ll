; RUN: rm -rf %t && mkdir -p %t/joint %t/jprof
; RUN: cp %S/../Inputs/fma_site_profile/tester.fpprofile %t/jprof/tester_a.fpprofile
; RUN: cp %S/../Inputs/fma_site_profile/tester.fpprofile %t/jprof/tester_b.fpprofile
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/piecewise_rewrite %opt < %S/../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-early-prune -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-rewrite-regime-search-factor=1024 -poseidon-cache=%t/noop -S -o /dev/null 2>&1 | FileCheck --check-prefix=NOOP --implicit-check-not="only frontier point" %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/one_rewrite_per_output %opt < %S/../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-early-prune -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/step -S -o /dev/null 2>&1 | FileCheck --check-prefix=STEP --implicit-check-not="only frontier point" %s
; RUN: sed 's/^fma,double,512$/fma,double,448/' %cost_models/cm_sm_120_synthetic.csv > %t/joint/cm_sm_120_fma.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.herbie.rkt %t/joint/cm_sm_120_fma.herbie.rkt
; RUN: cp %S/../Inputs/fma_opt_cache/herbie_default_958ABA9C1BC0AE00.json %t/joint/herbie_cuda-sm120_45B4DF20BF2EFFF8.json
; RUN: %opt < %S/../Inputs/no_early_prune/joint.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-print -poseidon-profile-use=%t/jprof -poseidon-early-prune -poseidon-enable-rewrite=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-precision=false -poseidon-cost-model=%t/joint/cm_sm_120_fma.csv -poseidon-solve-jointly -poseidon-cache=%t/joint -S -o /dev/null 2>&1 | FileCheck --check-prefix=JOINT --implicit-check-not="only frontier point" %s

; A DP frontier of one point is either the code as written, which no
; candidate improves on (NOOP), or a solution that is no costlier and no less
; accurate than the code as written and every other combination: one rewrite
; of the same cost and better accuracy (STEP). The note names which. A joint
; frontier of several points, the two sites' binary64 rewrites with and
; without their cheaper binary32 ones, has no note (JOINT).

; NOOP: Critical computation cost range: [0.000000e+00, 0.000000e+00]{{$}}
; NOOP-NEXT: DP table contains 1 entries.{{$}}
; NOOP-NEXT: [poseidon] none of the 1 candidate(s), alone or combined, is cheaper or more accurate than the code as written, which is the only frontier point.{{$}}
; NOOP-NEXT: Total candidate compositions: 2.000000e+00
; NOOP-NEXT: [poseidon] the no-op solution (cost budget used 0) is Pareto-optimal{{$}}
; NOOP-NOT: Applying solution

; STEP: Critical computation cost range: [[[C:[-+.e0-9]+]], [[C]]]{{$}}
; STEP-NEXT: DP table contains 1 entries.{{$}}
; STEP-NEXT: [poseidon] the only frontier point is a solution of 1 step(s) from the 2 candidate(s); it is no costlier and no less accurate than the code as written and every other combination.{{$}}
; STEP-NOT: no-op solution
; STEP: Computation cost budget used: [[C]]{{$}}
; STEP: [poseidon] Applying 1 step(s){{$}}

; JOINT: Critical computation cost range: [-7.480469e-01, -5.859375e-02]{{$}}
; JOINT-NEXT: DP table contains 3 entries.{{$}}
; JOINT-NEXT: Total candidate compositions:
