; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/site -poseidon-print -poseidon-show-table -S -o /dev/null 2>&1 | FileCheck --check-prefix=TABLE %s
; RUN: FileCheck --check-prefix=BUDGETS %s < %t/site/budgets.txt
; RUN: for b in $(tr , ' ' < %t/site/budgets.txt); do \
; RUN:   %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/site -poseidon-print -poseidon-cost-budget=$b -S -o /dev/null 2>&1 \
; RUN:     | grep -x "Computation cost budget used: $(printf %%e $b)" > /dev/null || { echo "budget $b does not select its point"; exit 1; }; \
; RUN: done
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/joint -poseidon-joint-dp -S -o /dev/null
; RUN: diff %t/site/budgets.txt %t/joint/budgets.txt
; RUN: for b in $(tr , ' ' < %t/joint/budgets.txt); do \
; RUN:   %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/joint -poseidon-joint-dp -poseidon-cost-budget=$b -S -o /dev/null 2>&1 \
; RUN:     | grep "^JOINT: minimum accuracy cost within budget: .*; computation cost used: $(printf %%e $b)$" > /dev/null || { echo "joint budget $b does not select its point"; exit 1; }; \
; RUN: done

; Every share budgets.txt lists, passed back as -poseidon-cost-budget, selects
; the frontier point it was written for, in a site's solve and in a joint one.
; The product of fp16_range_realization_nvptx.ll, whose operand A lies below
; FP16's smallest subnormal: the FP16 library call is the cheapest point of the
; frontier and carries the exponent penalty, so no budget selects it and
; budgets.txt does not list it.

; TABLE: Computation cost: -9.921875e-01, Accuracy cost: 1.000000e+30
; TABLE-NEXT: Solution steps:
; TABLE-NEXT: matmul[0] -> direct-dispatch cublas f16/f32

; BUDGETS-NOT: -0.9921875
; BUDGETS: {{^}}-0.98828125,
; BUDGETS-NOT: -0.9921875
