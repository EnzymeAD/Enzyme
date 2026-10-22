; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-apply-rewrites=R0_0 -poseidon-cache=%t -S 2>&1 | FileCheck %s
; REQUIRES: poseidon

; The rewritten clone is the code that ships, so its guard in front of a call
; that never returns has to survive canonicalization.

define double @tester(double %x, double %y) #0 {
entry:
  %bad = fcmp ole double %y, 0.000000e+00
  br i1 %bad, label %abort, label %ok

abort:
  call void @exit(i32 1)
  unreachable

ok:
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

declare void @exit(i32) noreturn nounwind
declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK-LABEL: define double @preprocess_tester
; CHECK-NOT: @llvm.assume
; CHECK: fcmp {{.*}}double %y, 0.000000e+00
; CHECK: call void @exit(i32 1)
; CHECK-NEXT: unreachable
; CHECK: ret double
