; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t -S 2>&1 | FileCheck %s
; REQUIRES: poseidon

; One marked body reached from TWO marker calls. The profiling run writes one
; record per BODY (profileSite shares one instrumented clone between a body's
; markers), while the solve clones the body once per call and LLVM uniques the
; second clone's name, so keying the profile on the clone would leave the
; second site with no profile and silently at its original precision. dFEM puts
; one integrator's marker into every registered kernel specialization, so this
; is the normal case there, not a corner one.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @site_a(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

define double @site_b(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; Both sites are served by the one profile the run recorded for @tester.
; CHECK-NOT: no profile at
; CHECK-NOT: left unchanged
; CHECK: [[BODY:preprocess_tester[^ :]*]]: tau=1.000000e-02
; CHECK: [[BODY2:preprocess_tester[^ :]*]]: tau=1.000000e-02
