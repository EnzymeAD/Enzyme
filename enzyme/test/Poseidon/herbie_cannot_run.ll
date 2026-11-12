; RUN: rm -rf %t && mkdir -p %t
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/no-herbie -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=ENABLED %s
; RUN: mkdir -p %t/tmp && not env TMPDIR=%t/tmp %clang -O1 -c -x ir %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/regime_split/profiles -mllvm -poseidon-enable-herbie=true -mllvm -poseidon-herbie-binary=%t/no-herbie -mllvm -poseidon-enable-pt=false -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv -mllvm -poseidon-cost-budget=1e9 -mllvm -poseidon-cache=%t/c -o %t/site.o 2>&1 | FileCheck --check-prefix=ENABLED --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: not ls %t/site.o
; RUN: ls %t/tmp | count 0
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=false -poseidon-herbie-binary=%t/no-herbie -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=DISABLED --implicit-check-not=Herbie %s

; A subgraph whose Herbie answer is not in the cache needs Herbie: when Herbie
; cannot be started the compile stops, naming the search, the cache and the
; binary, under opt and under clang alike: an error at the site, exit status
; 1, no object and none of clang's crash reporting. With algebraic rewriting
; off nothing asks Herbie, and the site is solved without it.

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

; ENABLED: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in
; ENABLED: error: {{.*}}Poseidon: the cache {{.*}}c holds no Herbie answer for subgraph 0 of preprocess_tester, and the Herbie at {{.*}}no-herbie cannot run ({{.+}}). Point -poseidon-herbie-binary at a Herbie, replay the answers another Herbie gave with -poseidon-import-herbie-answers, or turn algebraic rewriting off with -poseidon-enable-herbie=0.

; DISABLED: Solving the DP for preprocess_tester.
