; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/literal_rational %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t -S 2>&1 | FileCheck %s

; Herbie writes -5e-303 as a 17-digit numerator over a 318-digit denominator.
; The denominator alone overflows a double, so converting each half before
; dividing gives -0.0, and the regime split then sends y = 0 to the other arm.

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

; CHECK: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (if.f64
; CHECK: define double @preprocess_tester
; CHECK: fcmp ole double %y, -5.000000e-303
