; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/default -o %t/default.ll 2> %t/default.err
; RUN: FileCheck --check-prefix=NOSPLIT %s < %t/default.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/split -poseidon-min-uses-split=3 -poseidon-min-ops-split=4 -o %t/split.ll 2> %t/split.err
; RUN: FileCheck --check-prefix=SPLIT %s < %t/split.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/uses4 -poseidon-min-uses-split=4 -poseidon-min-ops-split=4 -o %t/uses4.ll 2> %t/uses4.err
; RUN: FileCheck --check-prefix=NOSPLIT %s < %t/uses4.err
; RUN: diff %t/default.ll %t/uses4.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/ops5 -poseidon-min-uses-split=3 -poseidon-min-ops-split=5 -o %t/ops5.ll 2> %t/ops5.err
; RUN: FileCheck --check-prefix=NOSPLIT %s < %t/ops5.err
; RUN: diff %t/default.ll %t/ops5.ll
; REQUIRES: poseidon
; Chain splitting (-poseidon-min-uses-split / -poseidon-min-ops-split, isExpansionBottleneck in Utils.cpp):
; %t has three users inside the FP subgraph and four operations (%a %b %c %t) upstream of it used by nothing
; else, so the one 9-operation subgraph splits at %t into a 4-operation unit and a 5-operation unit exactly
; when both thresholds are met (3 uses, 4 ops). One more use or one more operation demanded leaves the
; subgraph whole, and the Herbie-off precision-tuning solve is then byte-identical to the default (99/99).
; With the split, each unit gets its own candidate set and the DP composes them (10 -> 30 compositions).
; Profile: this module through the host profiler (opt -passes=poseidon,enzyme,poseidon-finalize
; -poseidon-profile-generate, linked with libposeidon_profile.a and Inputs/chain_split/split_driver.c).

define void @tester(double %x, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %x
  %c = fmul double %b, %b
  %t = fsub double %c, %y
  %u1 = fmul double %t, %x
  %u2 = fadd double %t, %y
  %u3 = fdiv double %t, %x
  %r1 = fadd double %u1, %u2
  %r2 = fmul double %u3, %u2
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; NOSPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; NOSPLIT-NOT: Bottleneck:
; NOSPLIT: [poseidon] After splitting, have 1 subgraphs in preprocess_tester
; NOSPLIT: Initial AccuracyCost: 8.837436e-11
; NOSPLIT-NEXT: Initial ComputationCost: 3.600000e+01
; NOSPLIT: Total candidate compositions: 1.000000e+01
; NOSPLIT: Minimum accuracy cost within budget: 2.672498e-03
; NOSPLIT-NEXT: Computation cost budget used: -128
; NOSPLIT: Applying solution for CS: All FP64(80%) + FP32(20%) (#7)

; SPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; SPLIT: Bottleneck:   %t = fsub double %c, %y
; SPLIT-NEXT: Num of operations that would be moved: 4 (>=4)
; SPLIT-NEXT: Num of internal uses: 3 (>=3)
; SPLIT-NEXT: Operations that would be moved:
; SPLIT-NEXT: %t = fsub double %c, %y
; SPLIT-NEXT: %c = fmul double %b, %b
; SPLIT-NEXT: %b = fadd double %a, %x
; SPLIT-NEXT: %a = fmul double %x, %y
; SPLIT: === Splitting subgraph at bottleneck:   %t = fsub double %c, %y
; SPLIT-NEXT: New subgraph:
; SPLIT-NEXT: Inputs (2):
; SPLIT: Operations (4):
; SPLIT: Outputs (1):
; SPLIT-NEXT: %t = fsub double %c, %y
; SPLIT-NEXT: Remaining subgraph:
; SPLIT-NEXT: Inputs (3):
; SPLIT: Operations (5):
; SPLIT: Outputs (2):
; SPLIT: Final subgraphs after splitting: 2
; SPLIT-NEXT: [poseidon] After splitting, have 2 subgraphs in preprocess_tester
; SPLIT: Initial AccuracyCost: 5.730790e-11
; SPLIT-NEXT: Initial ComputationCost: 1.600000e+01
; SPLIT: Initial AccuracyCost: 1.809276e-10
; SPLIT-NEXT: Initial ComputationCost: 2.000000e+01
; SPLIT: Total candidate compositions: 3.000000e+01
; SPLIT: Minimum accuracy cost within budget: 4.744200e-03
; SPLIT-NEXT: Computation cost budget used: -128
; SPLIT: Applying solution for CS: All FP64(60%) + FP32(40%) (#3)
