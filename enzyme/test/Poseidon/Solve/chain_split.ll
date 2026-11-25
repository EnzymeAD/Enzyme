; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/chain_split -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=-0.05 -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/default -o %t/default.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: %{opt} -poseidon-cache=%t/split -poseidon-min-uses-split=3 -poseidon-min-ops-split=4 -o %t/split.ll 2>&1 | FileCheck --check-prefix=SPLIT %s
; RUN: %{opt} -poseidon-cache=%t/uses4 -poseidon-min-uses-split=4 -poseidon-min-ops-split=4 -o %t/uses4.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: diff %t/default.ll %t/uses4.ll
; RUN: %{opt} -poseidon-cache=%t/ops5 -poseidon-min-uses-split=3 -poseidon-min-ops-split=5 -o %t/ops5.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: diff %t/default.ll %t/ops5.ll

; NOSPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; NOSPLIT-NOT: Bottleneck:
; NOSPLIT: [poseidon] After splitting, have 1 subgraphs in preprocess_tester
; NOSPLIT: Initial AccuracyCost: 8.837436e-11
; NOSPLIT-NEXT: Initial ComputationCost: 3.600000e+01
; NOSPLIT: Total candidate compositions: 1.000000e+01
; NOSPLIT: Minimum accuracy cost within budget: 2.672498e-03
; NOSPLIT-NEXT: Computation cost budget used: -5.555556e-02
; NOSPLIT: Applying solution for precision change All FP64(80%) + FP32(20%) (#7)

; SPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; SPLIT: Bottleneck:   %t = fsub double %c, %y
; SPLIT-NEXT: Num of operations that would be moved: 4 (>=4)
; SPLIT-NEXT: Num of internal uses: 3 (>=3)
; SPLIT-NEXT: Operations that would be moved:
; SPLIT-NEXT: %t = fsub double %c, %y
; SPLIT-NEXT: %c = fmul double %b, %b
; SPLIT-NEXT: %b = fadd double %a, %x
; SPLIT-NEXT: %a = fmul double %x, %y
; SPLIT: [poseidon] After splitting, have 2 subgraphs in preprocess_tester
; SPLIT: Initial AccuracyCost: 5.730790e-11
; SPLIT-NEXT: Initial ComputationCost: 1.600000e+01
; SPLIT: Initial AccuracyCost: 2.614053e-11
; SPLIT-NEXT: Initial ComputationCost: 2.000000e+01
; SPLIT: Total candidate compositions: 3.000000e+01
; SPLIT: Minimum accuracy cost within budget: 2.672498e-03
; SPLIT-NEXT: Computation cost budget used: -5.555556e-02
; SPLIT: Applying solution for precision change All FP64(60%) + FP32(40%) (#3)

; Chain splitting (-poseidon-min-uses-split, -poseidon-min-ops-split;
; isExpansionBottleneck in Solve/Subgraphs.cpp): %t has three users inside the
; FP subgraph and four operations (%a %b %c %t) upstream of it used by nothing
; else, so the 9-operation subgraph splits at %t into a 4-operation and a
; 5-operation unit exactly when both thresholds are met (3 uses, 4 ops).
; Demanding one more use or one more operation leaves the subgraph whole, and
; the solve then emits the default's module byte for byte. With the split each
; unit gets its own candidate set and the DP composes them (10 -> 30
; compositions).
; Profile: this module through the host profiler (opt
; -passes=poseidon,enzyme,poseidon-finalize -poseidon-profile-generate, linked
; with libposeidon_profile.a), driven by
;   for (int i = 0; i < 64; ++i) {
;     double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25;
;     double o1, o2, d1 = 1.0, d2 = 1.0;
;     site(x, y, &o1, &d1, &o2, &d2);
;   }

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
