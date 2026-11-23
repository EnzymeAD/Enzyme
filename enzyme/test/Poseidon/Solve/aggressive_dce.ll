; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/aggressive_dce -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.1 -poseidon-print -S -poseidon-cache=%t/keep 2>&1 | FileCheck --check-prefix=KEEP %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/aggressive_dce -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.1 -poseidon-print -S -poseidon-cache=%t/dce -poseidon-aggressive-dce 2>&1 | FileCheck --check-prefix=DCE %s
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
; KEEP: Poseidon: sensitivity floor raised the accuracy weight of 1 of 3 profiled instruction(s) in preprocess_tester to 1.000000e-06 x the site maximum sensitivity
; KEEP-NOT: Aggressive DCE
; KEEP: Applying solution for CS: All FP64(40%) + FP32(60%) (#1)
; KEEP: [poseidon] pin: tester:PT0=All FP64(40%) + FP32(60%)
; KEEP: ; ModuleID
; KEEP: define void @preprocess_tester(
; KEEP-NEXT: entry:
; KEEP-NEXT: [[XF:%.+]] = fptrunc {{.*}}double %x to float
; KEEP-NEXT: [[YF:%.+]] = fptrunc {{.*}}double %y to float
; KEEP-NEXT: [[ADD:%.+]] = fadd fast float [[YF]], [[XF]]{{$}}
; KEEP-NEXT: [[ADDD:%.+]] = fpext {{.*}}float [[ADD]] to double
; KEEP-NEXT: %keep = fmul fast double %x, [[ADDD]],
; KEEP-NEXT: store double %keep, ptr %o1
; KEEP-NEXT: [[DEAD:%.+]] = fmul fast float [[ADD]], [[YF]]{{$}}
; KEEP-NEXT: [[DEADD:%.+]] = fpext float [[DEAD]] to double
; KEEP-NEXT: store double [[DEADD]], ptr %o2
; KEEP-NEXT: ret void

; DCE: Aggressive DCE: eliminating zero-gradient non-critical instruction:   %dead = fmul fast double %add, %y, !enzyme_active
; DCE: [poseidon] After aggressive DCE, have 1 subgraphs in preprocess_tester
; DCE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; DCE: [poseidon] pin: tester:PT0=All FP64(0%) + FP32(100%)
; DCE: ; ModuleID
; DCE: define void @preprocess_tester(
; DCE-NEXT: entry:
; DCE-NEXT: [[XF:%.+]] = fptrunc {{.*}}double %x to float
; DCE-NEXT: [[YF:%.+]] = fptrunc {{.*}}double %y to float
; DCE-NEXT: [[ADD:%.+]] = fadd fast float [[YF]], [[XF]]{{$}}
; DCE-NEXT: [[KP:%.+]] = fmul fast float [[ADD]], [[XF]]{{$}}
; DCE-NEXT: [[KPD:%.+]] = fpext float [[KP]] to double
; DCE-NEXT: store double [[KPD]], ptr %o1
; DCE-NOT: ptr %o2
; DCE: ret void
