; RUN: rm -rf %t && mkdir -p %t/c
; RUN: printf '#!/bin/sh\nkill -KILL $$\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S 2>&1 | FileCheck --check-prefix=CONTINUE %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -poseidon-strict-mode -S -o /dev/null 2>&1 | FileCheck --check-prefix=STRICT %s
; RUN: ls %t/c | FileCheck --allow-empty --check-prefix=FILES %s

; A Herbie that crashes leaves its subgraph without algebraic candidates: the
; site is still solved, and here keeps its body as written. Under
; -poseidon-strict-mode the crash is fatal. Neither compile caches anything for
; the core.

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

; CONTINUE: [poseidon] Herbie subgraph 0 of preprocess_tester: crashed (signal: {{.+}}) after {{.*}} s.
; CONTINUE: Solving the DP for preprocess_tester.
; CONTINUE-NOT: Applying solution
; CONTINUE: [poseidon] no rewrite applied for preprocess_tester

; STRICT: error: {{.*}}Poseidon: strict mode: Herbie crashed (signal: {{.+}}) on subgraph 0 of preprocess_tester

; FILES-NOT: cachedHerbie
