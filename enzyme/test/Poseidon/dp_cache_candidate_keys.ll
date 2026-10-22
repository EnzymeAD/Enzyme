; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-print -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=SOLVE %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-print -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefix=HIT %s
; REQUIRES: poseidon

; The replayed Herbie result lists a regime split first and its exact else
; arm second. Pricing drops the split (its then arm is unreachable in the
; profiled box), so the solve picks the second candidate at position 0 of
; the surviving list. A compile that reuses the table skips pricing and sees
; both candidates; it has to apply the same rewrite, not position 0.

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
; SOLVE: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1)
; SOLVE: define double @preprocess_tester
; SOLVE-NOT: select
; SOLVE: fdiv
; SOLVE-NOT: select
; SOLVE: ret double

; HIT: Loaded DP tables from cache
; HIT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; HIT: define double @preprocess_tester
; HIT-NOT: select
; HIT: fdiv
; HIT-NOT: select
; HIT: ret double
