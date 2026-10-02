; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s --check-prefix=GEN
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/canonical_hash -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache= -S 2>&1 | FileCheck %s --check-prefix=MATCH
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/canonical_hash_missing -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache= -S 2>&1 | FileCheck %s --check-prefix=NOFIELD
; RUN: sed '/^define double @tester/,/^}/s|%mul = fmul fast double %add, %z|%extra = fmul fast double %add, %add\n  %mul = fmul fast double %extra, %z|' %s > %t.drift.ll
; RUN: not --crash %opt < %t.drift.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/canonical_hash -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache= -S 2>&1 | FileCheck %s --check-prefix=DRIFT
; REQUIRES: poseidon, enzyme

; The profile's slot indices are positions in the canonicalized clone's
; optimizable instruction sequence, so the profile carries a digest of that
; sequence and profile-use refuses a body that no longer produces it.

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

; The digest the profile-gen compile embeds is the one the fixture carries; the
; site id in front of it is what the condition-number probe arms by.
; GEN: @fpprofile_static_preprocess_tester = private unnamed_addr constant [45 x i8] c"SiteId = 0\0ACanonicalHash = f9f2bd660c90d989\0A\00"

; MATCH-NOT: has no CanonicalHash field
; MATCH-NOT: recorded against a different canonical form

; NOFIELD: Warning: {{.*}}canonical_hash_missing{{.*}} has no CanonicalHash field; the canonical form of preprocess_tester is not verified against the profile

; DRIFT: Poseidon: the profile of preprocess_tester was recorded against a different canonical form (profile CanonicalHash f9f2bd660c90d989, this compile {{[0-9a-f]+}})
