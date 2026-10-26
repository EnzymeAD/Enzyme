; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/expm1div_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%herbie_binary -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache= -poseidon-print -S 2>&1 | FileCheck %s
; REQUIRES: poseidon, herbie

; Live Herbie on (exp(x) - 1) / x: verifies expm1 candidate is produced.

declare double @llvm.exp.f64(double)

define double @tester(double %x) #0 {
entry:
  %exp = call fast double @llvm.exp.f64(double %x)
  %sub = fsub fast double %exp, 1.0
  %div = fdiv fast double %sub, %x
  ret double %div
}

define double @test_opt(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__poseidon_fp_optimize(double (double)* nonnull @tester, double %x, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: Candidates:
; CHECK: expm1
; CHECK: Finished
; CHECK: define double @preprocess_tester(double %x)
; CHECK: ret double
