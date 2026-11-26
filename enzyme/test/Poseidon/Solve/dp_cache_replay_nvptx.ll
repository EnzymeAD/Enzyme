; DEFINE: %{reprice} = env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie \
; DEFINE:   %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes=poseidon \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; DEFINE:   -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-0.9 -poseidon-print -poseidon-print-dp-table -S
; DEFINE: %{two} = %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes=poseidon \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv \
; DEFINE:   -poseidon-cost-budget=-0.99 -poseidon-print -poseidon-print-dp-table -S
; DEFINE: %{fms} = %opt %S/../Techniques/Precision/expansion_two_components_nvptx.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/expansion_two_components -poseidon-early-prune -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-1e-9 -poseidon-print -S
; RUN: rm -rf %t && mkdir %t

; A compile that reads its DP table back from the cache applies what the solve
; that wrote the table applied.

; A table that holds every kind of step. The exact solve of
; Inputs/no_early_prune/reprice.ll puts algebraic, precision and product steps on its
; frontier; the cache entry names an algebraic or precision step's candidate by key
; and a product step's by position. The second compile reads the table, binds
; every step to its unit again and prints the same table; the selected point (a
; precision change and a raise) is applied as on the solve, and the module is
; the solve's. The same for Inputs/no_early_prune/two_matmuls.ll, two products on one
; loop. The poseidon pass runs alone: the precision change calls libdevice float
; functions this hand-written module does not define, which poseidon-finalize
; refuses.
; RUN: mkdir %t/kinds
; RUN: %{reprice} -poseidon-cache=%t/kinds/reprice -o %t/kinds/solve.ll > %t/kinds/solve.log 2>&1
; RUN: %{reprice} -poseidon-cache=%t/kinds/reprice -o %t/kinds/replay.ll > %t/kinds/replay.log 2>&1
; RUN: FileCheck --check-prefix=KINDS-SOLVE %s < %t/kinds/solve.log
; RUN: FileCheck --check-prefix=KINDS-REPLAY %s < %t/kinds/replay.log
; RUN: awk '/^\[poseidon\] DP table:$/ {t = 1; print; next} t && /^ / {print; next} {t = 0}' %t/kinds/solve.log > %t/kinds/solve.table
; RUN: awk '/^\[poseidon\] DP table:$/ {t = 1; print; next} t && /^ / {print; next} {t = 0}' %t/kinds/replay.log > %t/kinds/replay.table
; RUN: diff %t/kinds/solve.table %t/kinds/replay.table
; RUN: diff %t/kinds/solve.ll %t/kinds/replay.ll
; RUN: grep -h -B3 '"unit": "R"' %t/kinds/reprice/dp_*.json | FileCheck --check-prefix=KINDS-TABLE-R %s
; RUN: grep -h -B3 '"unit": "P"' %t/kinds/reprice/dp_*.json | FileCheck --check-prefix=KINDS-TABLE-P %s
; RUN: grep -h -B2 '"unit": "M"' %t/kinds/reprice/dp_*.json | FileCheck --check-prefix=KINDS-TABLE-M --implicit-check-not=candidate_key %s
; RUN: %{two} -poseidon-cache=%t/kinds/two -o %t/kinds/two_solve.ll > %t/kinds/two_solve.log 2>&1
; RUN: %{two} -poseidon-cache=%t/kinds/two -o %t/kinds/two_replay.ll > %t/kinds/two_replay.log 2>&1
; RUN: FileCheck --check-prefix=KINDS-TWO %s < %t/kinds/two_replay.log
; RUN: awk '/^\[poseidon\] DP table:$/ {t = 1; print; next} t && /^ / {print; next} {t = 0}' %t/kinds/two_solve.log > %t/kinds/two_solve.table
; RUN: awk '/^\[poseidon\] DP table:$/ {t = 1; print; next} t && /^ / {print; next} {t = 0}' %t/kinds/two_replay.log > %t/kinds/two_replay.table
; RUN: diff %t/kinds/two_solve.table %t/kinds/two_replay.table
; RUN: diff %t/kinds/two_solve.ll %t/kinds/two_replay.ll

; KINDS-SOLVE: Solving the DP for preprocess_matmul_body.
; KINDS-SOLVE: DP table for preprocess_matmul_body cached to {{.*}}dp_{{[0-9A-F]+}}.json
; KINDS-SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; KINDS-SOLVE-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; KINDS-REPLAY-NOT: Solving the DP
; KINDS-REPLAY: Cache entry for preprocess_matmul_body found. Loaded DP tables from cache.
; KINDS-REPLAY: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; KINDS-REPLAY-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; KINDS-REPLAY: [poseidon] pin: matmul_body:P0=All FP64(0%) + FP32(100%),matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32

; Each step entry of a kind, as grep -B keeps it: the lines before its unit.
; KINDS-TABLE-R: "candidate_key": "{{[0-9A-F]+}}",
; KINDS-TABLE-R-NEXT: "index": {{[0-9]+}},
; KINDS-TABLE-R-NEXT: "unit": "R"
; KINDS-TABLE-P: "candidate_key": "{{[0-9A-F]+}}",
; KINDS-TABLE-P-NEXT: "index": {{[0-9]+}},
; KINDS-TABLE-P-NEXT: "unit": "P"
; KINDS-TABLE-M: "candidate": {{[0-9]+}},
; KINDS-TABLE-M-NEXT: "index": {{[0-9]+}},
; KINDS-TABLE-M-NEXT: "unit": "M"

; KINDS-TWO: Cache entry for preprocess_matmul_body found. Loaded DP tables from cache.
; KINDS-TWO: Applying solution for matmul[1] -> direct-dispatch cublas f16/f32 (#10)

; The replay emits the module the solve emitted, byte for byte: pricing, which
; the replay skips, leaves no declaration of its own in the module (here an
; FP32 fmuladd only a priced candidate uses). The replay prints no price it did
; not compute, and the numbers both print are the same.
; RUN: mkdir %t/same
; RUN: %{fms} -poseidon-cache=%t/same/c -o %t/same/fresh.ll 2> %t/same/fresh.log
; RUN: %{fms} -poseidon-cache=%t/same/c -o %t/same/replay.ll 2> %t/same/replay.log
; RUN: diff %t/same/fresh.ll %t/same/replay.ll
; RUN: FileCheck --check-prefix=SAME-FRESH %s < %t/same/fresh.log
; RUN: FileCheck --check-prefix=SAME-REPLAY --implicit-check-not="AccuracyCost: nan" --implicit-check-not=unpriced %s < %t/same/replay.log
; RUN: grep -E 'budget used|Applying solution' %t/same/fresh.log > %t/same/fresh.picked
; RUN: grep -E 'budget used|Applying solution' %t/same/replay.log > %t/same/replay.picked
; RUN: diff %t/same/fresh.picked %t/same/replay.picked

; SAME-FRESH: DP table for preprocess_fms_body cached to
; SAME-FRESH: Computation cost budget used:

; SAME-REPLAY: [poseidon] preprocess_fms_body: candidates not priced; the DP table read from the cache holds the prices its solve used
; SAME-REPLAY: Cache entry for preprocess_fms_body found. Loaded DP tables from cache.
; SAME-REPLAY: Computation cost budget used:
