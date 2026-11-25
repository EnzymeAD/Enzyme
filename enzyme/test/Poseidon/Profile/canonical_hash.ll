; RUN: %if enzyme %{ %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s --check-prefix=GEN %}
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -poseidon-print 2>&1 | FileCheck %s --check-prefix=MATCH --implicit-check-not=error: --implicit-check-not=CanonicalHash
; RUN: sed '/^define double @tester/,/^}/s|%mul = fmul fast double %add, %z|%extra = fmul fast double %add, %add\n  %mul = fmul fast double %extra, %z|' %s > %t.drift.ll
; RUN: not %opt < %t.drift.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S 2>&1 | FileCheck %s --check-prefix=DRIFT

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

; The header the profile-gen compile registers for the site at startup carries,
; behind the site id, the digest the fixture holds.
; GEN: @poseidon_site_preprocess_tester = private unnamed_addr constant [7 x i8] c"tester\00"
; GEN: @fpprofile_static_preprocess_tester = private unnamed_addr constant [45 x i8] c"SiteId = 0\0ACanonicalHash = f9f2bd660c90d989\0A\00"
; GEN: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess_tester, ptr null }]
; GEN-LABEL: define internal void @poseidon_register_static_preprocess_tester()
; GEN-NEXT: entry:
; GEN-NEXT: call void @poseidonRegisterProfileStatic(ptr @poseidon_site_preprocess_tester, ptr @fpprofile_static_preprocess_tester)
; GEN-NEXT: ret void

; A body that still produces the digest is solved from the profile: S is the
; largest result the fixture records, and the site calls the body it kept.
; MATCH-NOT: recorded against a different canonical form
; MATCH: [poseidon] preprocess_tester: tau=5.000000e-01 S=1.500000e+02 A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; MATCH-LABEL: define double @test_opt(
; MATCH-NEXT: entry:
; MATCH-NEXT: [[R:%.*]] = call double @tester(double %x, double %y, double %z)
; MATCH-NEXT: ret double [[R]]

; DRIFT: error: <unknown>:0:0: in function preprocess_tester {{.*}}: Poseidon: the profile of preprocess_tester was recorded against a different canonical form (profile CanonicalHash f9f2bd660c90d989, this compile {{[0-9a-f]+}}). Its slot indices do not name the same instructions; re-run profile generation.
