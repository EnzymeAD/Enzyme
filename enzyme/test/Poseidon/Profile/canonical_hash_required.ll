; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash_missing -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null > %t.log 2>&1; echo "exit=$?" >> %t.log
; RUN: FileCheck %s < %t.log

; A profile that does not say which canonical form its slot indices number
; cannot be checked against this body, so profile use refuses it.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: {{.*}}canonical_hash_missing{{.*}} has no CanonicalHash field
; CHECK-NOT: is not verified against the profile
; CHECK: exit={{[1-9][0-9]*}}
