; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: sed '1i CanonicalHash = 1fa032681b79bf6d' %S/../Inputs/regime_split/profiles/preprocess_tester.fpprofile > %t/prof/preprocess_tester.fpprofile
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -poseidon-report-path=%t/solve -S -o /dev/null 2>&1 | FileCheck --check-prefix=SOLVE %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/c -poseidon-report-path=%t/hit -S -o /dev/null 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: FileCheck --check-prefix=REWRITES %s < %t/solve/preprocess_tester_rewrites.json
; RUN: sed -n 's/^ *"id": "\(R[0-9]*_[0-9]*\)",$/\1/p' %t/solve/preprocess_tester_rewrites.json > %t/solve.id
; RUN: FileCheck --check-prefix=EXPLAIN -DID=$(cat %t/solve.id) %s < %t/solve/preprocess_tester.explain.json
; RUN: FileCheck --check-prefix=EXPLAIN -DID=$(cat %t/solve.id) %s < %t/hit/preprocess_tester.explain.json

; Pricing drops the regime split Herbie lists first and a compile that reads
; the table back prices nothing. The rewrite id the solve's _rewrites.json
; gives the applied rewrite is the id both compiles' explain reports give it.

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

; SOLVE: Solving the DP for preprocess_tester.

; HIT: Loaded DP tables from cache

; REWRITES: "id": "R{{[0-9]+}}_{{[0-9]+}}",
; REWRITES: "rewritten_expr": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"
; REWRITES: "total_rewrites": 1

; EXPLAIN: "result": {
; EXPLAIN-NEXT: "applied": [
; EXPLAIN-NEXT: "[[ID]]"
; EXPLAIN: "fpcore": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))",
; EXPLAIN-NOT: "id":
; EXPLAIN: "id": "[[ID]]",
