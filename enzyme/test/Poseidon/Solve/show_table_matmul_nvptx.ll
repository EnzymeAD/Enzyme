; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cost-budget=0 -poseidon-print -poseidon-show-table -poseidon-cache=%t -S -o /dev/null 2>&1 | FileCheck %s

; Under a cost budget the matrix product's candidates are in the DP table, and
; -poseidon-show-table prints each frontier point's matrix step as the solution
; it applies is printed: the product, the option, its position.

; CHECK: *** DP Table ***
; CHECK: Computation cost: {{-?[0-9.e+-]+}}, Accuracy cost: 1.144101e-06
; CHECK-NEXT: Solution steps:
; CHECK-NEXT: matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7)
; CHECK: Computation cost: {{-?[0-9.e+-]+}}, Accuracy cost: 0.000000e+00
; CHECK-NEXT: Solution steps:
; CHECK-NEXT: matmul[0] -> native-dgemm cublas f64/f64 (#13)
; CHECK-NEXT: *** End of DP Table ***
; CHECK: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
