; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=true \
; DEFINE:   -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir -p %t/legacy %t/fixed %t/wide
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/legacy -poseidon-rewrite-min-regime-samples=0 2>&1 \
; RUN:   | FileCheck --check-prefix=LEGACY %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/fixed 2>&1 | FileCheck --check-prefix=FIXED %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/piecewise_rewrite %{opt} -poseidon-cache=%t/wide -poseidon-rewrite-regime-search-factor=1024 2>&1 \
; RUN:   | FileCheck --check-prefix=WIDE %s

; sqrt(x*x + y*y) - x over x in [1, 2], y in [-1, 1]. The replayed Herbie
; result splits at y <= -0.9998 and returns 0 there, which is wrong by ~0.4;
; the other arm is the exact rationalised form. The arm covers 1e-4 of the
; profiled box, so none of the 1024 base samples reaches it and, priced on the
; base samples alone, the split looks more accurate than the original.

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
