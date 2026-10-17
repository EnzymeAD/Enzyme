; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/one_rewrite_per_output/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t -S 2>&1 | FileCheck %s
; REQUIRES: poseidon

; The binary64 and binary32 searches of one output return the same exact
; rewrite. Each search is its own item, so a solve that may take one rewrite
; from each counts the rewrite's accuracy gain twice; only one is ever
; materialized, and the solution may name only one.

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

; CHECK: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0))
; CHECK-NOT: Applying solution for
; CHECK: !!! Solution applied !!!
; CHECK: define double @preprocess_tester
; CHECK: hypot
