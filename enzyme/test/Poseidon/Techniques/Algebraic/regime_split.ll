; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-print -S
; RUN: rm -rf %t && mkdir -p %t/legacy %t/fixed %t/wide %t/literal && split-file %s %t/literal-herbie
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/legacy -poseidon-rewrite-min-regime-samples=0 2>&1 \
; RUN:   | FileCheck --check-prefix=LEGACY %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/fixed 2>&1 | FileCheck --check-prefix=FIXED %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/wide -poseidon-rewrite-regime-search-factor=1024 2>&1 \
; RUN:   | FileCheck --check-prefix=WIDE %s
; RUN: env FAKE_HERBIE_DIR=%t/literal-herbie %{opt} -poseidon-cache=%t/literal 2>&1 | FileCheck --check-prefix=LITERAL %s

; sqrt(x*x + y*y) - x over x in [1, 2], y in [-1, 1], with Herbie's regime split
; at y <= -0.9998 replayed: that arm returns 0, wrong by ~0.4, and covers 1e-4
; of the profiled box, so priced on the 1024 base samples alone the split looks
; better than the original (LEGACY); an arm no sample reaches drops the split
; (FIXED), and a wider search prices the arm and rejects it (WIDE).
; Herbie writes -5e-303 as a 17-digit numerator over a 318-digit denominator,
; which alone overflows a double: the literal is converted as one rational, not
; half by half, so y = 0 stays in its arm (LITERAL).

; LEGACY-NOT: regime-split candidate
; LEGACY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (if.f64
; LEGACY: define double @preprocess_tester
; LEGACY: select

; FIXED: [poseidon] regime-split candidate: arms (samples:cand/orig mean error) else:1024:{{.*}} then:{{[0-9]}}:{{.*}} -> dropped (unpriced arm)
; FIXED-NOT: Applying solution
; FIXED: define double @preprocess_tester
; FIXED-NOT: select
; FIXED: ret double

; WIDE: [poseidon] regime-split candidate: arms (samples:cand/orig mean error) else:1024:{{.*}} then:32:{{[0-9.]+}}e-01/{{.*}} -> {{[0-9.]+}}e-01;
; WIDE-NOT: Applying solution
; WIDE: define double @preprocess_tester
; WIDE-NOT: select
; WIDE: ret double

; LITERAL: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (if.f64
; LITERAL: define double @preprocess_tester
; LITERAL: fcmp ole double %y, -5.000000e-303

;--- results.json
{"tests":[{"bits":64,"name":"0","status":"imp-start","prec":"binary64","output":"(if.f64 (<=.f64 v1 #s(literal -1930258305619341/386051661123868214325895970762095083331216144111904370034983364157543830047598546775742309000849007597326427200921653578548066591998660043462778854257084865420374725869305346230443778499781067545394454342790117394565596548890481374012190543459242928201313126587598361115137891035519604744312911050121319319358268243968 binary64)) (/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))","cost-accuracy":[[1.0,30.0],[1.5,0.5,"(if.f64 (<=.f64 v1 #s(literal -1930258305619341/386051661123868214325895970762095083331216144111904370034983364157543830047598546775742309000849007597326427200921653578548066591998660043462778854257084865420374725869305346230443778499781067545394454342790117394565596548890481374012190543459242928201313126587598361115137891035519604744312911050121319319358268243968 binary64)) (/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))"],[]]}]}
