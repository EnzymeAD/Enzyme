; DEFINE: %{opt} = %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles \
; DEFINE:   -poseidon-enable-herbie=true \
; DEFINE:   -poseidon-herbie-processes=1 -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh 2>&1 | FileCheck --check-prefix=FRESH %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/no_alternatives.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/no_bits.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NO-BITS %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-BITS-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/not_object.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NOT-OBJECT %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NOT-OBJECT-STRICT %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/unknown_core.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=UNKNOWN-CORE %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/zero_alternatives.json %t/c/cachedHerbieOutput_*
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=ZERO-ALTERNATIVES %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/approx_without_result.json %t/c/cachedHerbieOutput_*
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=APPROX %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/literal_binary16.json %t/c/cachedHerbieOutput_*
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=LITERAL %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/operator_f16.json %t/c/cachedHerbieOutput_*
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=OPERATOR %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/unknown_operator.json %t/c/cachedHerbieOutput_*
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=UNKNOWN-OPERATOR %s
; RUN: cp %S/../Inputs/herbie_malformed_answer/operator_arity.json %t/c/cachedHerbieOutput_*
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=ARITY %s

; The answers are replayed from the cache with no Herbie to run. A cached
; answer that is not shaped as Herbie writes its results gives the
; subgraph no algebraic candidates, as a crash of Herbie does: the site is
; solved without them, and under -poseidon-strict-mode the compile stops. An
; answer that lists no alternatives offers Herbie's best expression alone. An
; expression the model cannot read, malformed or with an operator, format or
; operand count it does not model, stops the compile with an error at the site
; naming the answer's cache file, before anything evaluates or lowers it.

; FRESH: Saved Herbie output to cache file

; NO-ALTERNATIVES: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"). No algebraic candidates for this subgraph from it.
; NO-ALTERNATIVES: warning: {{.*}}Poseidon returned false (no change) for preprocess_tester{{$}}
; NO-ALTERNATIVES-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"). Without -poseidon-strict-mode the subgraph is left as written.

; NO-BITS: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (missing value at tests[0].bits, core "0"). No algebraic candidates for this subgraph from it.
; NO-BITS: warning: {{.*}}Poseidon returned false (no change) for preprocess_tester{{$}}
; NO-BITS-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (missing value at tests[0].bits, core "0").

; NOT-OBJECT: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (expected an object with a "tests" array at the top level). No algebraic candidates for this subgraph from it.
; NOT-OBJECT: warning: {{.*}}Poseidon returned false (no change) for preprocess_tester{{$}}
; NOT-OBJECT-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected an object with a "tests" array at the top level).

; UNKNOWN-CORE: [poseidon] Herbie subgraph 0 of preprocess_tester: the cached answer is not a Herbie results file (tests[0] names core "2", and the cores asked about are 0 to 1). No algebraic candidates for this subgraph from it.

; ZERO-ALTERNATIVES: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1)

; APPROX: error: <unknown>:0:0: in function preprocess_tester double (double, double): Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}/c/cachedHerbieOutput_default_{{[0-9A-F]+}}.txt has, for core "0", an approx form without its specification and result in '#s(approx v0)'; Poseidon cannot model that expression. Remove the file to search again, or turn algebraic rewriting off with -poseidon-enable-herbie=0.{{$}}
; LITERAL: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt has, for core "0", a literal of the unmodelled format binary16 in '#s(literal 1 binary16)'; Poseidon cannot model
; OPERATOR: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt has, for core "0", the operator +.f16 of an unmodelled format in '(+.f16 v0 v1)'; Poseidon cannot model
; UNKNOWN-OPERATOR: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt has, for core "0", the operator erfc, which Poseidon does not model in '(erfc.f64 v0)'; Poseidon cannot model
; ARITY: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt has, for core "0", the operator < with 3 operand(s), not 2 in '(<.f64 v0 v1 v0)'; Poseidon cannot model
