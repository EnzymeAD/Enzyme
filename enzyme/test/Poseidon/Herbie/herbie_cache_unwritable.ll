; RUN: rm -rf %t && mkdir -p %t && touch %t/file
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/file/c -S -o /dev/null 2>&1 | FileCheck %s

; A Herbie answer that cannot be cached, here because the cache directory is
; under a regular file, stops the compile naming the path.

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

; CHECK: error: {{.*}}preprocess_tester{{.*}}: Poseidon: cannot create the Herbie cache directory {{.*}}file/c: Not a directory. Herbie's answers are kept in the -poseidon-cache directory, which has to be writable.
