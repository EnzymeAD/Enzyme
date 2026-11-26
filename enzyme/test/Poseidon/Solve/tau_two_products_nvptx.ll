; RUN: rm -rf %t && mkdir %t
; RUN: %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-tau=1e-3 -poseidon-cache=%t/c -poseidon-explain=%t/x -poseidon-print -S -o /dev/null 2>&1 | FileCheck %s
; RUN: %python -c "import json, sys; r = json.load(open(sys.argv[1]))['result']; print(r['pin'], '%%.6e' %% r['modelled_rel_error'])" %t/x/matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s
; DEFINE: %{off} = %opt %S/../Inputs/no_early_prune/two_matmuls.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/no_early_prune/two_matmuls -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-print -S
; RUN: %{off} -poseidon-cost-budget=-0.5 -poseidon-cache=%t/budget -poseidon-explain=%t/off -o %t/budget.ll 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not='Applying solution' %s
; RUN: FileCheck --check-prefix=OFF-IR --input-file=%t/budget.ll %s
; RUN: %python -c "import json, sys; [print(u['id'], a['status'], a['reason']) for u in json.load(open(sys.argv[1]))['subgraphs'] for a in u['alternatives']]" %t/off/matmul_body.explain.json | FileCheck --check-prefix=OFF-EXPLAIN %s
; RUN: %{off} -poseidon-cost-budget=-0.5 -poseidon-early-prune -poseidon-cache=%t/budget-pruned -o /dev/null 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not='Applying solution' %s
; RUN: %{off} -poseidon-tau=1e-3 -poseidon-cache=%t/tau -o /dev/null 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not='Applying solution' %s
; RUN: %{off} -poseidon-tau=1e-3 -poseidon-early-prune -poseidon-cache=%t/tau-pruned -o /dev/null 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not='Applying solution' %s

; Two matrix products on one reduction loop (Inputs/no_early_prune/two_matmuls.ll):
; a raise of either replaces the loop, so no solution takes both. Under a
; target the products are units of the one DP, which keeps them apart: one of
; them runs as a library call, and the site's error is that product's. The two
; calls cost the same and model the same error to the last digits, so either
; may be the one.

; CHECK-NOT: second matrix product
; CHECK: [poseidon] preprocess_matmul_body: tau=1.000000e-03 {{.*}} rel=3.137966e-04{{$}}
; CHECK: [poseidon] Applying 1 step(s)
; CHECK-NEXT: [poseidon] Applying solution for matmul[{{[01]}}] -> direct-dispatch cublas f16/f32 (#7){{$}}
; CHECK-NOT: second matrix product
; EXPLAIN: matmul_body:M{{[01]}}=direct-dispatch cublas f16/f32 3.137966e-04{{$}}

; An in-kernel raise takes the place of the whole loop, which also computes the
; other product, so neither product is offered one. With library calls off, no
; candidate is left, under a budget or a target, and both products stay as
; written.
; OFF-COUNT-2: [poseidon] preprocess_matmul_body: the reduction loop computes a value besides the product that is used after it, so it has no in-kernel raise{{$}}
; OFF-IR-LABEL: define dso_local void @matmul_body(
; OFF-IR: store double [[D:%[0-9]+]],
; OFF-IR-NEXT: getelementptr
; OFF-IR-NEXT: store double [[E:%[0-9]+]],
; OFF-IR: [[D]] = tail call double @llvm.fmuladd.f64(
; OFF-IR: [[E]] = tail call double @llvm.fmuladd.f64(
; OFF-EXPLAIN: M0 rejected not offered: the reduction loop computes a value besides the product that is used after it, and the raise takes the place of the whole loop{{$}}
; OFF-EXPLAIN-NEXT: M1 rejected not offered: the reduction loop computes a value besides the product that is used after it, and the raise takes the place of the whole loop{{$}}
