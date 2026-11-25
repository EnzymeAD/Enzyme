; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-print -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/site -poseidon-explain=%t/site/ex -poseidon-show-table 2>&1 | FileCheck --check-prefix=TABLE %s
; RUN: %poseidon_explain %t/site/ex --format budgets > %t/site/budgets
; RUN: FileCheck --check-prefix=BUDGETS %s < %t/site/budgets
; RUN: for b in $(tr , ' ' < %t/site/budgets); do \
; RUN:   %{opt} -poseidon-cache=%t/site -poseidon-cost-budget=$b 2>&1 \
; RUN:     | grep -x "Computation cost budget used: $(printf %%e $b)" > /dev/null || { echo "budget $b does not select its point"; exit 1; }; \
; RUN: done
; RUN: %{opt} -poseidon-cache=%t/joint -poseidon-explain=%t/joint/ex -poseidon-joint-dp
; RUN: %poseidon_explain %t/joint/ex --format budgets > %t/joint/budgets
; RUN: diff %t/site/budgets %t/joint/budgets
; RUN: for b in $(tr , ' ' < %t/joint/budgets); do \
; RUN:   %{opt} -poseidon-cache=%t/joint -poseidon-joint-dp -poseidon-cost-budget=$b 2>&1 \
; RUN:     | grep -x "Computation cost budget used: $(printf %%e $b)" > /dev/null || { echo "joint budget $b does not select its point"; exit 1; }; \
; RUN: done

; Every share the explanation lists as a point's budget, passed back as
; -poseidon-cost-budget, selects the frontier point it was written for, in a
; site's solve and in a joint one. The product of
; fp16_range_realization_nvptx.ll, whose operand A lies below FP16's smallest
; subnormal: the FP16 library call is the cheapest point of the frontier and
; carries the exponent penalty, so no budget selects it and it lists none.

; TABLE: Computation cost: -9.921875e-01, Accuracy cost: 1.000000e+30
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: matmul[0] -> direct-dispatch cublas f16/f32

; BUDGETS-NOT: -0.9921875
; BUDGETS: {{^}}-0.98828125,
; BUDGETS-NOT: -0.9921875
