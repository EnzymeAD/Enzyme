; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/aggressive_dce -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/keep 2>&1 | FileCheck --check-prefix=KEEP %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/aggressive_dce -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -S -poseidon-cache=%t/dce -poseidon-aggressive-dce 2>&1 | FileCheck --check-prefix=DCE %s
; REQUIRES: poseidon
; One FP subgraph with two outputs; the profile seeds o1 with 1 and o2 with 0, so %dead has an exactly zero gradient. -poseidon-aggressive-dce deletes it before the solve; without the flag it keeps its floored weight and is rewritten. Profile from running this module through the host profiler with Inputs/aggressive_dce/dce_driver.c.

define void @tester(double %x, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %add = fadd fast double %x, %y
  %keep = fmul fast double %add, %x
  store double %keep, ptr %o1, align 8
  %dead = fmul fast double %add, %y
  store double %dead, ptr %o2, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; KEEP-NOT: Aggressive DCE
; KEEP: define void @preprocess_tester(
; KEEP: store double %{{.*}}, ptr %o1
; KEEP: store double %{{.*}}, ptr %o2
; KEEP: ret void

; DCE: Aggressive DCE: eliminating zero-gradient non-critical instruction:   %dead = fmul fast double %add, %y
; DCE: [poseidon] After aggressive DCE, have 1 subgraphs in preprocess_tester
; DCE: define void @preprocess_tester(
; DCE: store double %{{.*}}, ptr %o1
; DCE-NOT: ptr %o2
; DCE: ret void
