; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/literal_rational %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t -S -poseidon-print 2>&1 | FileCheck %s

; Herbie writes -5e-303 as a 17-digit numerator over a 318-digit denominator.
; The denominator alone overflows a double, so converting each half before
; dividing gives -0.0, and the regime split then sends y = 0 to the other arm.

; CHECK: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (if.f64
; CHECK: define double @preprocess_tester
; CHECK: fcmp ole double %y, -5.000000e-303
