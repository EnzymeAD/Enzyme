; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache= -S | FileCheck %s
; REQUIRES: poseidon

; Opt phase with no Herbie/PT: nothing is applied, so the wrapper calls the
; ORIGINAL body and the preprocess clone is dropped.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

; enzyme_not_overwritten is a modifier of the marker that follows it, exactly as
; Enzyme's own handleArguments decodes it, and it is accepted on either side of
; that marker. Decoding it as a marker of its own drops the activity it
; prefixes, and %y's shadow would then be read as a fourth argument.
define double @test_opt(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, metadata !"enzyme_not_overwritten", metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", metadata !"enzyme_not_overwritten", double %y, double %ys, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: define double @test_opt(double %x, double %y, double %z, double %xs, double %ys)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[call:.+]] = call double @tester(double %x, double %y, double %z)
; CHECK-NEXT:   ret double %[[call]]
; CHECK-NOT: @preprocess_tester

