; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie %opt %S/../Inputs/exact_dp/reprice.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.9 -poseidon-print -poseidon-show-table -poseidon-report-path=%t/rep -poseidon-explain=%t/ex -poseidon-cache=%t/exact -S -o %t/exact.ll 2>&1 | FileCheck --check-prefix=TABLE %s
; RUN: FileCheck --check-prefix=TEXT %s < %t/rep/preprocess_matmul_body.txt
; RUN: FileCheck --check-prefix=JSON %s < %t/rep/preprocess_matmul_body.json
; RUN: FileCheck --check-prefix=CURATED %s < %t/rep/preprocess_matmul_body_rewrites.json
; RUN: cat %t/ex/preprocess_matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie %opt %S/../Inputs/exact_dp/reprice.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau -S -o %t/tau.ll 2>&1 | FileCheck --check-prefix=TAU %s

; The table print, the reports of -poseidon-report-path and the explanation
; of a site with algebraic, precision and product units (the module of
; Inputs/exact_dp/reprice.ll, solved exactly; the budget selects a precision
; change and a raise). Under a target the products are selected by their
; domain error apart from the DP, and a precision change of the product's
; epilogue that the DP selects is dropped against the raise. The poseidon pass
; runs alone.

; TABLE: *** DP Table ***
; TABLE-NEXT: Computation cost: -9.537642e-01, Accuracy cost: 5.593938e+09
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: CS: All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; TABLE: Computation cost: -3.657315e-01, Accuracy cost: 7.434626e+03
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)-> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; TABLE-NEXT: CS: All FP64(0%) + FP32(100%) (#0)
; TABLE: *** End of DP Table ***
; TABLE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; TEXT: Pareto table: 21 points, cost range [-9.537642e-01, 1.454545e-01]
; TEXT-NEXT: Candidate outputs: 4, Candidate subgraphs: 2
; TEXT: Expression #0: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1)))
; {{[^ -~]+}} stands for the Greek capital delta the log prints.
; TEXT: Subgraph #0: 54 candidates ({{[^ -~]+}}cost, {{[^ -~]+}}acc; sorted by {{[^ -~]+}}cost)
; TEXT: Subgraph #1: 9 candidates ({{[^ -~]+}}cost, {{[^ -~]+}}acc; sorted by {{[^ -~]+}}cost)
; TEXT: --- Pareto Point #0: Cost=-9.537642e-01, Accuracy=5.593938e+09 ---
; TEXT-NEXT: [Precision] All FP64(0%) + FP32(100%) (#0)
; TEXT-NEXT: FP64 -> FP32 for 12 operations
; TEXT-NEXT: [Matmul] matmul[0] -> wmma m16n16k16 f16/f32 (#0)

; JSON: "candidate_outputs": [
; JSON-NEXT: {
; JSON-NEXT: "executions": 512,
; JSON-NEXT: "gradient": 7039721{{[0-9]{6}\.[0-9]+}},
; JSON-NEXT: "initial_accuracy_cost": 2856.889{{[0-9]+}},
; JSON-NEXT: "initial_computation_cost": 19456,
; JSON-NEXT: "num_candidates": 2,
; JSON-NEXT: "original_expr": "(+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1)))"
; JSON: "candidate_subgraphs": [
; JSON-NEXT: {
; JSON-NEXT: "candidates": [
; JSON-NEXT: {
; JSON-NEXT: "accuracy_cost_delta": 531561.5{{[0-9]+}},
; JSON-NEXT: "computation_cost_delta": -9819136,
; JSON-NEXT: "description": "All FP64(0%) + FP32(100%)",
; JSON-NEXT: "id": "PT0_0"
; JSON: "pareto_points": [
; JSON-NEXT: {
; JSON-NEXT: "accuracy_cost": 5593937{{[0-9]{3}\.[0-9]+}},
; JSON-NEXT: "computation_cost": -0.9537642045454545{{[0-9]*}},
; JSON-NEXT: "steps": [
; JSON-NEXT: {
; JSON-NEXT: "candidate_index": 0,
; JSON: "from": "FP64",
; JSON-NEXT: "num_operations": 12,
; JSON-NEXT: "source_locations": [],
; JSON-NEXT: "to": "FP32"
; JSON-NEXT: }
; JSON-NEXT: ],
; JSON-NEXT: "description": "All FP64(0%) + FP32(100%)",
; JSON-NEXT: "type": "precision_change"
; JSON-NEXT: },
; JSON-NEXT: {
; JSON-NEXT: "accuracy_cost_delta": 5593406{{[0-9]{3}\.[0-9]+}},
; JSON-NEXT: "candidate_index": 0,
; JSON-NEXT: "computation_cost_delta": -3932160,
; JSON-NEXT: "executions": 8192,
; JSON-NEXT: "matmul_id": 0,
; JSON-NEXT: "target": "wmma m16n16k16 f16/f32",
; JSON-NEXT: "type": "matmul_rewrite"
; JSON: "computation_cost": -0.3657315340909090{{[0-9]*}},
; JSON-NEXT: "steps": [
; JSON-NEXT: {
; JSON-NEXT: "accuracy_cost_delta": 88.41621{{[0-9]*}},
; JSON: "computation_cost_delta": -4880896,
; JSON-NEXT: "executions": 512,
; JSON-NEXT: "gradient": 7039721{{[0-9]{6}\.[0-9]+}},
; JSON-NEXT: "herbie_accuracy": 0.65625,
; JSON-NEXT: "herbie_cost": 0.5,
; JSON-NEXT: "initial_herbie_accuracy": 0.6875,
; JSON-NEXT: "original_expr": "(+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1)))",
; JSON-NEXT: "rewritten_expr": "(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1)))))",
; JSON-NEXT: "source_locations": [],
; JSON-NEXT: "type": "rewrite"

; CURATED: "rewrites": [
; CURATED-NEXT: {
; CURATED-NEXT: "accuracy_cost_delta": 0,
; CURATED-NEXT: "affected_instructions": [
; CURATED-NEXT: "  %as = tail call double @__nv_asin(double %w)
; CURATED-NEXT: ],
; CURATED-NEXT: "category": "free_win",
; CURATED-NEXT: "computation_cost_delta": 0,
; CURATED-NEXT: "description": "Funcs FP64(90%) + Expansion2(10%)",
; CURATED-NEXT: "efficiency": 0,
; CURATED-NEXT: "id": "PT0_31",
; CURATED-NEXT: "source_locations": [],
; CURATED-NEXT: "type": "precision_change"
; CURATED: "accuracy_cost_delta": 88.41621{{[0-9]*}},
; CURATED-NEXT: "category": "speed_for_accuracy",
; CURATED-NEXT: "computation_cost_delta": -4880896,
; CURATED-NEXT: "efficiency": 55203.63{{[0-9]*}},
; CURATED-NEXT: "executions": 512,
; CURATED-NEXT: "gradient": 7039721{{[0-9]{6}\.[0-9]+}},
; CURATED-NEXT: "herbie_accuracy": 0.65625,
; CURATED-NEXT: "id": "R0_1",
; CURATED-NEXT: "initial_herbie_accuracy": 0.6875,
; CURATED-NEXT: "original_expr": "(+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1)))",
; CURATED-NEXT: "rewritten_expr": "(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1)))))"
; CURATED: "total_rewrites": 65

; EXPLAIN: "result": {
; EXPLAIN-NEXT: "applied": [
; EXPLAIN-NEXT: "PT0_0",
; EXPLAIN-NEXT: "M0_2"
; EXPLAIN-NEXT: ]

; TAU: Poseidon error-budget: matmul -> tcec n=2 wmma m16n16k16 f16/f32 (domain err 1.576106e-07 <= 1.000000e-06 at confidence 9.500000e-01)
; TAU: [poseidon] preprocess_matmul_body: an elementwise rewrite the DP selected edits instructions a selected matrix-product raise consumes; keeping the raise and dropping the rewrite
; TAU: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; TAU-NEXT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; TAU: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32,matmul_body:PT0=All FP64(0%) + FP32(100%){{$}}
