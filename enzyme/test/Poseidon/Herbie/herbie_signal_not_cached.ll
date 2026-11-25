; RUN: rm -rf %t && mkdir -p %t/c
; RUN: printf '#!/bin/sh\n[ -z "$FAKE_HERBIE_SIGNAL" ] || kill -KILL $$\nexec %S/../Inputs/fake-herbie.sh "$@"\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=SIGNAL --implicit-check-not="KILLED after" --implicit-check-not="Saved Herbie output" %s
; RUN: ls %t/c | FileCheck --allow-empty --check-prefix=FILES %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=RETRY %s

; A Herbie process that dies from a signal (here SIGKILL, as the OOM killer
; sends) well inside its wall budget crashed; it did not time out. The compile
; says so and caches nothing, so the next compile runs Herbie again.

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

; SIGNAL: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout 1800 s, wall budget 2040 s, key {{[0-9A-F]+$}}
; SIGNAL-NEXT: [poseidon] Herbie subgraph 0 of preprocess_tester: crashed (signal: {{.+}}) after {{[0-9.e+-]+}} s. No algebraic candidates for this subgraph in this compile; nothing is cached, so the next compile runs Herbie again.{{$}}
; SIGNAL: DP table contains 1 entries.
; SIGNAL: [poseidon] no rewrite applied for preprocess_tester; calling original body tester (bit-transparent no-op)

; FILES-NOT: cachedHerbie

; RETRY-NOT: cached TIMEOUT
; RETRY: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout 1800 s, wall budget 2040 s, key [[KEY:[0-9A-F]+]]{{$}}
; RETRY: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_[[KEY]].txt{{$}}
; RETRY-NEXT: [poseidon] Herbie subgraph 0 of preprocess_tester: completed in {{[0-9.e+-]+}} s, a fresh answer of the Herbie with digest {{[0-9A-F]+}}{{$}}
; RETRY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; RETRY: [poseidon] Herbie answers: 1 fresh, 0 from the cache{{$}}
