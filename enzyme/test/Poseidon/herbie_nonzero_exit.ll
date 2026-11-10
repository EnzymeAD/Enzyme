; RUN: rm -rf %t && mkdir -p %t/c %t/none
; RUN: printf '#!/bin/sh\n%S/Inputs/fake-herbie.sh "$@"\nexit 1\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: printf '#!/bin/sh\nexit 3\n' > %t/no-results.sh && chmod +x %t/no-results.sh
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=USED --implicit-check-not=crash %s
; RUN: not --crash %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/no-results.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/none -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONE --implicit-check-not=crashed %s

; A Herbie that exits with a nonzero status is not a crash. When it wrote a
; usable results.json, the compile says so once and uses and caches the
; results; when it wrote none, the compile stops.

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

; USED: [poseidon] Warning: Herbie subgraph 0 of preprocess_tester exited with status 1 but wrote a usable results.json; using it.
; USED: Saved Herbie output to cache file
; USED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)

; NONE: Poseidon: Herbie exited with status 3 and produced no usable results.json for subgraph 0 of preprocess_tester
