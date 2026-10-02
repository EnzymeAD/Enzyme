; RUN: rm -rf %t && mkdir -p %t/legacy %t/fixed %t/wide
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/regime_split/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-print -poseidon-cache=%t/legacy -poseidon-min-arm-samples=0 -S 2>&1 | FileCheck --check-prefix=LEGACY %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/regime_split/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-print -poseidon-cache=%t/fixed -S 2>&1 | FileCheck --check-prefix=FIXED %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/regime_split/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-print -poseidon-cache=%t/wide -poseidon-arm-search-factor=1024 -S 2>&1 | FileCheck --check-prefix=WIDE %s
; REQUIRES: poseidon

; sqrt(x*x + y*y) - x over x in [1, 2], y in [-1, 1]. The replayed Herbie
; result splits at y <= -0.9998 and returns 0 there, which is wrong by ~0.4;
; the other arm is the exact rationalised form. The arm covers 1e-4 of the
; profiled box, so none of the 1024 base samples reaches it and, priced on the
; base samples alone, the split looks more accurate than the original.

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
