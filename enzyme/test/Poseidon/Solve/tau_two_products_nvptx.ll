; RUN: rm -rf %t && mkdir %t
; RUN: %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-3 -poseidon-cache=%t/c -poseidon-explain=%t/x -poseidon-print -S -o /dev/null 2>&1 | FileCheck %s
; RUN: %python -c "import json, sys; r = json.load(open(sys.argv[1]))['result']; print(r['pin'], '%%.6e' %% r['modelled_rel_error'])" %t/x/matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s

; Two matrix products on one reduction loop (Inputs/no_early_prune/two_matmuls.ll):
; a raise of either replaces the loop, so no solution takes both. Under a
; target the products are units of the one DP, which keeps them apart: one of
; them runs as a library call, and the site's error is that product's. The two
; calls cost the same and model the same error to the last digits, so either
; may be the one.

; CHECK-NOT: second matrix product
; CHECK: [poseidon] preprocess_matmul_body: tau=1.000000e-03 {{.*}} rel=3.137966e-04{{$}}
; CHECK: [poseidon] Applying 1 step(s)
; CHECK-NEXT: [poseidon] Applying solution for matmul[{{[01]}}] -> direct-dispatch cublas f16/f32 (#10){{$}}
; CHECK-NOT: second matrix product
; EXPLAIN: matmul_body:M{{[01]}}=direct-dispatch cublas f16/f32 3.137966e-04{{$}}
