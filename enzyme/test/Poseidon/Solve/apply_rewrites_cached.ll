; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-print -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=SOLVE %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-apply-rewrites=R0_1 -poseidon-print -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=APPLY %s

; The solve drops the regime split the replayed Herbie result lists first, and
; its report numbers the exact rewrite R0_1, its place in Herbie's list. A table
; is now cached; applying R0_1 by hand next to it has to name the same rewrite,
; although it is at position 0 of the list pricing leaves.

define double @tester(double %x, double %y) #0 {
entry:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @llvm.sqrt.f64(double %s)
  %d = fsub double %r, %x
  ret double %d
}

define double @test_opt(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; SOLVE: dropped (unpriced arm)
; SOLVE: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)

; APPLY: Selecting R0_1: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1)
; APPLY: define double @preprocess_tester
; APPLY-NOT: select
; APPLY: ret double
