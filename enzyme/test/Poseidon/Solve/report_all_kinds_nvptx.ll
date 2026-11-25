; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.9 -poseidon-print -poseidon-print-dp-table -poseidon-explain=%t/ex -poseidon-cache=%t/exact -S -o %t/exact.ll 2>&1 | FileCheck --check-prefix=TABLE %s
; RUN: %python -c "import json, sys; d = json.load(open(sys.argv[1])); print(len(d['pareto']), 'points;', *(u['id'] + ':' + str(len(u['alternatives'])) for u in d['subgraphs'])); [print(p['budget'], ','.join(p['applied'])) for p in d['pareto'][:2]]" %t/ex/matmul_body.explain.json | FileCheck --check-prefix=SUMMARY %s
; RUN: cat %t/ex/matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau -S -o %t/tau.ll 2>&1 | FileCheck --check-prefix=TAU %s

; The table print and the explanation of a site with algebraic, precision and
; product units (the module of Inputs/no_early_prune/reprice.ll, solved exactly; the
; budget selects a precision change and a raise). Under a target the products are selected by their
; domain error apart from the DP, and a precision change of the product's
; epilogue that the DP selects is dropped against the raise. The poseidon pass
; runs alone.

; TABLE: [poseidon] DP table:
; TABLE-NEXT: computation cost -9.537642e-01, accuracy cost 5.593938e+09, steps:
; TABLE-NEXT: precision change All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; TABLE: computation cost -3.657315e-01, accuracy cost 7.434626e+03, steps:
; TABLE-NEXT: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)-> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; TABLE-NEXT: precision change All FP64(0%) + FP32(100%) (#0)
; TABLE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; SUMMARY: 21 points; S0:56 S1:9 M0:3
; SUMMARY-NEXT: -0.9537642045454545 P0_0,M0_0
; SUMMARY-NEXT: -0.94921875 P0_0,M0_1

; EXPLAIN: "result": {
; EXPLAIN-NEXT: "applied": [
; EXPLAIN-NEXT: "P0_0",
; EXPLAIN-NEXT: "M0_2"
; EXPLAIN-NEXT: ]
; EXPLAIN: "cost_delta": -{{0\.[0-9]+}},
; EXPLAIN-NEXT: "fpcore": "(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1)))))",
; EXPLAIN-NEXT: "herbie_error_bits": {
; EXPLAIN-NEXT: "after": 22,
; EXPLAIN-NEXT: "before": 20
; EXPLAIN-NEXT: },
; EXPLAIN-NEXT: "id": "R0_1",

; TAU: [poseidon] error budget: matmul -> tcec n=2 wmma m16n16k16 f16/f32 (domain err 1.576106e-07 <= 1.000000e-06 at confidence 9.500000e-01)
; TAU: warning: {{.*}}preprocess_matmul_body{{.*}}: Poseidon: a selected rewrite edits instructions a selected matrix product consumes; the rewrite is dropped{{$}}
; TAU: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; TAU-NEXT: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; TAU: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32,matmul_body:P0=All FP64(0%) + FP32(100%){{$}}
