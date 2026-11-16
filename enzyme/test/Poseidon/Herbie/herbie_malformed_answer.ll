; DEFINE: %{opt} = %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles \
; DEFINE:   -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh \
; DEFINE:   -poseidon-herbie-processes=1 -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} 2>&1 | FileCheck --check-prefix=FRESH %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/no_alternatives.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES %s
; RUN: not %{opt} -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/no_bits.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} 2>&1 | FileCheck --check-prefix=NO-BITS %s
; RUN: not %{opt} -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-BITS-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/not_object.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} 2>&1 | FileCheck --check-prefix=NOT-OBJECT %s
; RUN: not %{opt} -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NOT-OBJECT-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/unknown_core.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} 2>&1 | FileCheck --check-prefix=UNKNOWN-CORE %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/zero_alternatives.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=ZERO-ALTERNATIVES %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/approx_without_result.json %t/c/cachedHerbieOutput_*
; RUN: not --crash %{opt} 2>&1 | FileCheck --check-prefix=APPROX %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/literal_binary16.json %t/c/cachedHerbieOutput_*
; RUN: not --crash %{opt} 2>&1 | FileCheck --check-prefix=LITERAL %s

; A cached answer that is not shaped as Herbie writes its results gives the
; subgraph no algebraic candidates, as a crash of Herbie does: the site is
; solved without them, and under -poseidon-strict-mode the compile stops. An
; answer that lists no alternatives offers Herbie's best expression alone. An
; expression that does not parse stops the compile.

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

; FRESH: Saved Herbie output to cache file

; NO-ALTERNATIVES: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"). No algebraic candidates for this subgraph from it.
; NO-ALTERNATIVES: [poseidon] no rewrite applied for preprocess_tester
; NO-ALTERNATIVES-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"). Without -poseidon-strict-mode the subgraph is left as written.

; NO-BITS: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (missing value at tests[0].bits, core "0"). No algebraic candidates for this subgraph from it.
; NO-BITS: [poseidon] no rewrite applied for preprocess_tester
; NO-BITS-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (missing value at tests[0].bits, core "0").

; NOT-OBJECT: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (expected an object with a "tests" array at the top level). No algebraic candidates for this subgraph from it.
; NOT-OBJECT: [poseidon] no rewrite applied for preprocess_tester
; NOT-OBJECT-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected an object with a "tests" array at the top level).

; UNKNOWN-CORE: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (tests[0] names core "2", and the cores asked about are 0 to 1). No algebraic candidates for this subgraph from it.

; ZERO-ALTERNATIVES: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1)

; APPROX: LLVM ERROR: Malformed approx expression: #s(approx v0)
; LITERAL: LLVM ERROR: Herbie expr parser: Unexpected constant dtype: binary16
