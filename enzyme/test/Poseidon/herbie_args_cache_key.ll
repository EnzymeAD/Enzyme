; RUN: rm -rf %t && mkdir -p %t
; RUN: printf '#!/bin/sh\nexec %S/Inputs/dp_cache_keys/fake-herbie.sh "$@"\n' > %t/other-herbie.sh && chmod +x %t/other-herbie.sh
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=FIRST %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -poseidon-herbie-num-iters=2 -S -o /dev/null 2>&1 | FileCheck --check-prefix=OTHER %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=BACK %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/dp_cache_keys/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -poseidon-herbie-timeout=50 -poseidon-herbie-num-threads=2 -S -o /dev/null 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/other-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=BINARY %s
; REQUIRES: poseidon

; A Herbie result is cached under its cores, the binary, the search arguments
; and the platform, and a DP table records them. Changing a search argument
; searches again and re-solves; going back finds the first search's results,
; but the table was last solved under the other argument, so it is solved
; again. Herbie's timeouts and worker count decide no result, so changing them
; reads both caches. Another Herbie binary searches again and re-solves.

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

; FIRST: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; FIRST-NOT: DP cache entry
; FIRST: Solving the DP for preprocess_tester.

; OTHER-NOT: Using cached Herbie output
; OTHER: Saved Herbie output to cache file
; OTHER: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-herbie-num-iters; re-solving.

; BACK-NOT: Saved Herbie output
; BACK: Using cached Herbie output from {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; BACK: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-herbie-num-iters; re-solving.

; HIT-NOT: Saved Herbie output
; HIT: Using cached Herbie output from
; HIT-NOT: DP cache entry
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.

; BINARY-NOT: Using cached Herbie output
; BINARY: Saved Herbie output to cache file
; BINARY: [poseidon] DP cache entry for preprocess_tester was solved for a different Herbie binary; re-solving.
