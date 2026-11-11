; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.9 -poseidon-print -poseidon-show-table -poseidon-cache=%t/reprice -S -o %t/solve.ll > %t/solve.log 2>&1
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.9 -poseidon-print -poseidon-show-table -poseidon-cache=%t/reprice -S -o %t/replay.ll > %t/replay.log 2>&1
; RUN: FileCheck --check-prefix=SOLVE %s < %t/solve.log
; RUN: FileCheck --check-prefix=REPLAY %s < %t/replay.log
; RUN: sed -n '/^\*\*\* DP Table/,/^\*\*\* End of DP Table/p' %t/solve.log > %t/solve.table
; RUN: sed -n '/^\*\*\* DP Table/,/^\*\*\* End of DP Table/p' %t/replay.log > %t/replay.table
; RUN: diff %t/solve.table %t/replay.table
; RUN: diff %t/solve.ll %t/replay.ll
; RUN: grep -B3 '"itemType": "CO"' %t/reprice/table.json | FileCheck --check-prefix=TABLE-CO %s
; RUN: grep -B3 '"itemType": "CS"' %t/reprice/table.json | FileCheck --check-prefix=TABLE-CS %s
; RUN: grep -B2 '"itemType": "CM"' %t/reprice/table.json | FileCheck --check-prefix=TABLE-CM --implicit-check-not=candidateKey %s
; RUN: %opt %S/exact_dp_two_matmuls_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/two_matmuls -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.99 -poseidon-print -poseidon-show-table -poseidon-cache=%t/two -S -o %t/two_solve.ll > %t/two_solve.log 2>&1
; RUN: %opt %S/exact_dp_two_matmuls_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/two_matmuls -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-early-prune=false -poseidon-cost-budget=-0.99 -poseidon-print -poseidon-show-table -poseidon-cache=%t/two -S -o %t/two_replay.ll > %t/two_replay.log 2>&1
; RUN: FileCheck --check-prefix=TWO %s < %t/two_replay.log
; RUN: sed -n '/^\*\*\* DP Table/,/^\*\*\* End of DP Table/p' %t/two_solve.log > %t/two_solve.table
; RUN: sed -n '/^\*\*\* DP Table/,/^\*\*\* End of DP Table/p' %t/two_replay.log > %t/two_replay.table
; RUN: diff %t/two_solve.table %t/two_replay.table
; RUN: diff %t/two_solve.ll %t/two_replay.ll

; A DP table that holds every kind of step, read back. The exact solve of
; exact_dp_reprice_nvptx.ll puts algebraic, precision and product steps on its
; frontier; table.json names an algebraic or precision step's candidate by key
; and a product step's by position. The second compile reads the table, binds
; every step to its unit again and prints the same table; the selected point
; (a precision change and a raise) is applied as on the solve, and the module
; is the solve's. The same for
; exact_dp_two_matmuls_nvptx.ll, two products on one loop. The poseidon pass
; runs alone: the precision change calls libdevice float functions this
; hand-written module does not define, which poseidon-finalize refuses.

; SOLVE: Solving the DP for preprocess_matmul_body.
; SOLVE: DP table for preprocess_matmul_body cached to file.
; SOLVE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; SOLVE-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; REPLAY-NOT: Solving the DP
; REPLAY: Cache entry for preprocess_matmul_body found. Loaded DP tables from cache.
; REPLAY: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; REPLAY-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; REPLAY: [poseidon] pin: matmul_body:PT0=All FP64(0%) + FP32(100%),matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32

; TABLE-CO: "candidateKey": "{{[0-9A-F]+}}",
; TABLE-CO-NEXT: "itemIndex": {{[0-9]+}},
; TABLE-CO-NEXT: "itemType": "CO"
; TABLE-CS: "candidateKey": "{{[0-9A-F]+}}",
; TABLE-CS-NEXT: "itemIndex": {{[0-9]+}},
; TABLE-CS-NEXT: "itemType": "CS"
; TABLE-CM: "candidateIndex": {{[0-9]+}},
; TABLE-CM-NEXT: "itemIndex": {{[0-9]+}},
; TABLE-CM-NEXT: "itemType": "CM"

; TWO: Cache entry for preprocess_matmul_body found. Loaded DP tables from cache.
; TWO: Applying solution for matmul[1] -> direct-dispatch cublas f16/f32 (#11)
