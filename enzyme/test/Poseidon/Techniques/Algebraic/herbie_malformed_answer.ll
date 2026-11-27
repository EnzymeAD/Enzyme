; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=true \
; DEFINE:   -poseidon-herbie-processes=1 -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -poseidon-print -S -o /dev/null
; RUN: rm -rf %t && split-file %s %t/answers
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh 2>&1 | FileCheck --check-prefix=FRESH %s
; RUN: cp %t/answers/no_alternatives.json %t/c/herbie_*.json
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-ALTERNATIVES-STRICT %s
; RUN: cp %t/answers/no_bits.json %t/c/herbie_*.json
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NO-BITS %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NO-BITS-STRICT %s
; RUN: cp %t/answers/not_object.json %t/c/herbie_*.json
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=NOT-OBJECT %s
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=NOT-OBJECT-STRICT %s
; RUN: cp %t/answers/unknown_core.json %t/c/herbie_*.json
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=UNKNOWN-CORE %s
; RUN: cp %t/answers/zero_alternatives.json %t/c/herbie_*.json
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=ZERO-ALTERNATIVES %s
; RUN: cp %t/answers/approx_without_result.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=APPROX %s
; RUN: cp %t/answers/literal_binary16.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=LITERAL %s
; RUN: cp %t/answers/operator_f16.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=OPERATOR %s
; RUN: cp %t/answers/unknown_operator.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=UNKNOWN-OPERATOR %s
; RUN: cp %t/answers/operator_arity.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=ARITY %s
; RUN: cp %t/answers/zero_denominator.json %t/c/herbie_*.json
; RUN: not %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=ZERO-DENOMINATOR %s

; Cached answers replayed with no Herbie to run (split-file sections below). One
; not shaped as Herbie writes its results gives no algebraic candidate, fatal
; under -poseidon-strict-mode; one listing no alternatives offers Herbie's best
; alone; an expression the model cannot read stops the compile naming the file.

; FRESH: [poseidon] Saved Herbie output to cache file

; NO-ALTERNATIVES: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"), so the subgraph has no algebraic candidate from it{{$}}
; NO-ALTERNATIVES: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}
; NO-ALTERNATIVES-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected [initial, best, alternatives] at tests[0].cost-accuracy, core "0"); without -poseidon-strict-mode the subgraph is left as written{{$}}

; NO-BITS: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (missing value at tests[0].bits, core "0"), so the subgraph has no algebraic candidate from it{{$}}
; NO-BITS: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}
; NO-BITS-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (missing value at tests[0].bits, core "0"); without -poseidon-strict-mode the subgraph is left as written{{$}}

; NOT-OBJECT: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected an object with a "tests" array at the top level), so the subgraph has no algebraic candidate from it{{$}}
; NOT-OBJECT: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}
; NOT-OBJECT-STRICT: error: {{.*}}Poseidon: strict mode: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (expected an object with a "tests" array at the top level); without -poseidon-strict-mode the subgraph is left as written{{$}}

; UNKNOWN-CORE: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: the cached answer for subgraph 0 of preprocess_tester is not a Herbie results file (tests[0] names core "2", and the cores asked about are 0 to 1), so the subgraph has no algebraic candidate from it{{$}}

; ZERO-ALTERNATIVES: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1)

; APPROX: error: <unknown>:0:0: in function preprocess_tester double (double, double): Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (an approx form without its specification and result in '#s(approx v0)'); remove the file to search again, or turn algebraic rewriting off with -poseidon-enable-algebraic=0{{$}}
; LITERAL: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (a literal of the unmodelled format binary16 in '#s(literal 1 binary16)')
; OPERATOR: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (the operator +.f16 of an unmodelled format in '(+.f16 v0 v1)')
; UNKNOWN-OPERATOR: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (the operator erfc, which Poseidon does not model in '(erfc.f64 v0)')
; ARITY: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (the operator < with 3 operand(s), not 2 in '(<.f64 v0 v1 v0)')
; ZERO-DENOMINATOR: error: {{.*}}Poseidon: the Herbie answer for subgraph 0 of preprocess_tester in {{.*}}herbie_default_{{[0-9A-F]+}}.json has, for core "0", an expression Poseidon cannot model (a rational literal with denominator 0 in '#s(literal 1/0 binary64)')

;--- approx_without_result.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "#s(approx v0)", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
;--- literal_binary16.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(+.f64 v0 #s(literal 1 binary16))", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
;--- no_alternatives.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6, "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"]]}]}}
;--- no_bits.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"name": "0", "status": "imp-start", "output": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6, "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"], []]}]}}
;--- not_object.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": []}
;--- operator_arity.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(if (<.f64 v0 v1 v0) v0 v1)", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
;--- operator_f16.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(+.f16 v0 v1)", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
;--- unknown_core.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"name": "2", "status": "error", "output": "#f"}]}}
;--- unknown_operator.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(+.f64 (erfc.f64 v0) v1)", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
;--- zero_alternatives.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6, "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"], []]}]}}
;--- zero_denominator.json
{"key": {"Herbie binary": "0000000000000000", "format": "1"}, "value": {"tests": [{"bits": 64, "name": "0", "status": "imp-start", "output": "(+.f64 v0 #s(literal 1/0 binary64))", "cost-accuracy": [[1.0, 30.0], [1.4, 0.6], []]}]}}
