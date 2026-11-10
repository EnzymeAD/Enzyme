; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))" -S 2>&1 | FileCheck --check-prefix=NAMED %s
; RUN: not env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=R0=(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) #s(literal 0 binary64) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))" -S -o /dev/null 2>&1 | FileCheck --check-prefix=DROPPED %s
; RUN: not env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=R0=(-.f64 v0 v1)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=UNKNOWN %s

; The value a replay from the cached table prints applies what the replay
; applied.
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c -poseidon-cost-budget=1e9 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SOLVE %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c -poseidon-cost-budget=1e9 -S -o %t/replay.ll 2> %t/replay.err
; RUN: FileCheck --check-prefix=REPLAY %s < %t/replay.err
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/replay.err | paste -sd, -)" -S -o %t/pinned.ll 2> %t/pinned.err
; RUN: diff %t/replay.ll %t/pinned.ll
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/replay.err > %t/replay.picks
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/pinned.err > %t/pinned.picks
; RUN: diff %t/replay.picks %t/pinned.picks

; -poseidon-apply-rewrites names an algebraic rewrite by its expression. The
; regime split Herbie lists first is a name of no candidate, because pricing
; dropped it, and that is what the error says. Module and Herbie answer are
; apply_rewrites_cached.ll's.

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

; NAMED: [poseidon] Selecting tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)): (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1)
; NAMED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; NAMED: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; NAMED-LABEL: define double @preprocess_tester
; NAMED-NOT: select
; NAMED: ret double

; DROPPED: LLVM ERROR: [poseidon] tester: R0 has no candidate named '(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) {{.*}}': pricing dropped it, because a regime-split arm is reached by fewer than {{[0-9]+}} samples

; UNKNOWN: LLVM ERROR: [poseidon] tester: R0 has no candidate named '(-.f64 v0 v1)'. Its candidates are:
; UNKNOWN-NEXT: {{^}}  (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}

; SOLVE: DP table for preprocess_tester cached to file.
; SOLVE: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}

; REPLAY: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; REPLAY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; REPLAY: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
