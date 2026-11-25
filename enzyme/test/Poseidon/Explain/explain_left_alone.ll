; RUN: rm -rf %t && mkdir %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-explain=%t/ex -S -o /dev/null 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: FileCheck --check-prefix=JSON %s < %t/ex/preprocess_counter.explain.json
; RUN: %poseidon_explain %t/ex | FileCheck --check-prefix=REPORT %s

; A site the pass leaves as written before any solve has an explanation too,
; which says what kept it from the solve.

; WARN: warning: {{.*}}counter holds no floating-point operation the pass rewrites; leaving the site unchanged

; JSON: "function": {
; JSON-NEXT: "ir": "preprocess_counter",
; JSON-NEXT: "name": "counter",
; JSON-NEXT: "symbol": "counter"
; JSON-NEXT: },
; JSON-NEXT: "result": {
; JSON-NEXT: "reason": "the site holds no floating-point operation the pass rewrites",
; JSON-NEXT: "status": "no_fp_operation"
; JSON-NEXT: },
; JSON-NEXT: "schema": "poseidon-explain/2",
; JSON-NEXT: "target": {
; JSON-NEXT: "tau": 0.5,
; JSON-NEXT: "tau_from": "site"
; JSON-NEXT: }

; REPORT: Poseidon site counter  [counter]
; REPORT-NEXT: accuracy target tau = 0.5 (the site's own); no fp operation: the site holds no floating-point operation the pass rewrites

define i64 @counter(i64 %n) {
entry:
  %r = add i64 %n, 1
  ret i64 %r
}

define i64 @site(i64 %n) {
entry:
  %0 = tail call i64 (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @counter, i64 %n, metadata !"poseidon_tau", double 0.5)
  ret i64 %0
}

declare i64 @__poseidon_fp_optimize(ptr, ...)
