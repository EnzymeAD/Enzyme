; RUN: rm -rf %t && mkdir -p %t/prof %t/c
; RUN: sed '1i CanonicalHash = 1fa032681b79bf6d' %S/Inputs/regime_split/profiles/preprocess_tester.fpprofile > %t/prof/preprocess_tester.fpprofile
; RUN: printf '#!/bin/sh\n[ -z "$FAKE_HERBIE_SIGNAL" ] || kill -KILL $$\nexec %S/Inputs/fake-herbie.sh "$@"\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -S -o /dev/null > %t/signal.log 2>&1 || true
; RUN: FileCheck --check-prefix=SIGNAL --implicit-check-not="KILLED after" %s < %t/signal.log
; RUN: ls %t/c | FileCheck --allow-empty --check-prefix=FILES %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=RETRY %s

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

; SIGNAL: Herbie{{.*}}{{[Cc]rash}}

; FILES-NOT: cachedHerbieTimeout_

; RETRY-NOT: cached TIMEOUT
; RETRY: Saved Herbie output to cache file
; RETRY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)
