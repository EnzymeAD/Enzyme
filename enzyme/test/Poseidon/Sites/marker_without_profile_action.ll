; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_profile(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y, metadata !"poseidon_tau", double 1.0e-6)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

; CHECK: define double @test_profile(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fadd:.+]] = call double @tester(double %x, double %y)
; CHECK-NEXT:   ret double %[[fadd]]
; CHECK-NEXT: }
