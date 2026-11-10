; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-early-prune=false -poseidon-comp-cost-budget=-1328475425 -poseidon-print -poseidon-show-table -poseidon-report-path=%t/rep -poseidon-explain=%t/ex -poseidon-cache=%t/exact -S -o %t/exact.ll > %t/exact.log 2>&1
; RUN: FileCheck --check-prefix=TABLE %s < %t/exact.log
; RUN: FileCheck --check-prefix=TEXT %s < %t/rep/preprocess_matmul_body.txt
; RUN: FileCheck --check-prefix=JSON %s < %t/rep/preprocess_matmul_body.json
; RUN: FileCheck --check-prefix=CURATED %s < %t/rep/preprocess_matmul_body_rewrites.json
; RUN: cat %t/ex/preprocess_matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau -S -o %t/tau.ll > %t/tau.log 2>&1
; RUN: FileCheck --check-prefix=TAU %s < %t/tau.log
; REQUIRES: poseidon, scev-trunc-iv-nowrap

; The table print, the reports of -poseidon-report-path and the explanation
; of a site with algebraic, precision and product units
; (exact_dp_reprice_nvptx.ll, solved exactly; the budget selects a precision
; change and a raise). Under a target the products are selected by their
; domain error apart from the DP, and a precision change of the product's
; epilogue that the DP selects is dropped against the raise. The poseidon pass
; runs alone, as in dp_cache_replay_all_kinds_nvptx.ll.

; TABLE: *** DP Table ***
; TABLE-NEXT: Computation cost: -1340878898, Accuracy cost: 5.593938e+09
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: CS: All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; TABLE: Computation cost: -537702278, Accuracy cost: 7.434626e+03
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; TABLE: *** End of DP Table ***
; TABLE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; TABLE-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; TEXT: Pareto table: 21 points, cost range [-1340878898, 186944845]
; TEXT-NEXT: Candidate outputs: 4, Candidate subgraphs: 2
; TEXT: Expression #0: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1)))
; TEXT: Subgraph #0: 54 candidates (Δcost, Δacc; sorted by Δcost)
; TEXT: Subgraph #1: 9 candidates (Δcost, Δacc; sorted by Δcost)
; TEXT: --- Pareto Point #0: Cost=-1340878898, Accuracy=5.593938e+09 ---
; TEXT-NEXT: [Precision] All FP64(0%) + FP32(100%) (#0)
; TEXT-NEXT: FP64 -> FP32 for 12 operations
; TEXT-NEXT: [Matmul] matmul[0] -> wmma m16n16k16 f16/f32 (#0)

; JSON: "candidate_outputs": [
; JSON: "candidate_subgraphs": [
; JSON: "type": "precision_change"
; JSON: "type": "matmul_rewrite"

; CURATED: "id": "PT0_31",
; CURATED: "total_rewrites": 65

; EXPLAIN: "result": {
; EXPLAIN-NEXT: "applied": [
; EXPLAIN-NEXT: "PT0_0",
; EXPLAIN-NEXT: "M0_2"
; EXPLAIN-NEXT: ]

; TAU: Poseidon error-budget: matmul -> tcec n=2 wmma m16n16k16 f16/f32
; TAU: [poseidon] preprocess_matmul_body: an elementwise rewrite the DP selected edits instructions a selected matrix-product raise consumes; keeping the raise and dropping the rewrite
; TAU: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; TAU-NEXT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; TAU: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32,matmul_body:PT0=All FP64(0%) + FP32(100%)
