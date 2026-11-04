; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -poseidon-cache=%t/base -S -o /dev/null 2> %t/base.err
; RUN: FileCheck --check-prefix=BASE %s < %t/base.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-100 -poseidon-print -poseidon-cache=%t/split -poseidon-split-calls=pow -S -o /dev/null 2> %t/split.err
; RUN: FileCheck --check-prefix=SPLIT %s < %t/split.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/pow_term/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-1 -poseidon-print -poseidon-split-calls=pow -poseidon-cache=%t/herbie -S -o %t/herbie.ll 2> %t/herbie.err
; RUN: FileCheck --check-prefix=HERBIE-CORE %s < %t/herbie.err
; RUN: FileCheck --check-prefix=HERBIE %s < %t/herbie.ll
; REQUIRES: poseidon

; A rate term as generated reacting-flow code writes it (Exasim's reacting-air
; source has nine): pow(T, fma(-log T, a, b)), scaled to about 1e-25, with log T
; also used by the second output.
;
; -poseidon-split-calls=pow gives the pow call its own FP subgraph with the
; operations only it uses (the fma and the fneg; log T stays with its other
; user), whatever -poseidon-min-uses-split/-poseidon-min-ops-split (99 here) say.
;
; The accuracy model draws the leaves the inputs are computed from (T, and y)
; and evaluates the inputs from them, so T and log T are drawn consistently; here
; every draw lands inside the profiled input box.
;
; The pow subgraph reads T and l = log T. Its Herbie core spells l as (log v0),
; its definition over T, so Herbie can use pow(T, y) = exp(y log T). A (log v0)
; in a returned rewrite maps back to %l: the fake Herbie's exact rewrite
; exp(fma(-log T, a, b) log T) is priced without a second log (Delta CompCost
; 256, not 768), and its pow(T, -1.62) is applied.
;
; profiles: this module through the host profiler (opt
; -passes=poseidon,enzyme,poseidon-finalize -poseidon-profile-generate, linked with
; libposeidon_profile_host.a, libposeidon_profile.a and Inputs/pow_term/driver.c).

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %r1 = fmul double %k, %y
  %h = fmul double %nl, %y
  %r2 = fadd double %h, %T
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %T, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %T, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; BASE: Found a subgraph with 8 operations and 2 inputs and 2 outputs
; BASE-NOT: Bottleneck:
; BASE: [poseidon] After splitting, have 1 subgraphs in preprocess_tester

; SPLIT: Bottleneck:   %p = call double @llvm.pow.f64(double %T, double %e)
; SPLIT: === Splitting subgraph at bottleneck:   %p = call double @llvm.pow.f64(double %T, double %e)
; SPLIT-NEXT: New subgraph:
; SPLIT: Operations (3):
; SPLIT-NEXT: %p = call double @llvm.pow.f64
; SPLIT-NEXT: %e = call double @llvm.fma.f64
; SPLIT-NEXT: %nl = fneg double %l
; SPLIT: Remaining subgraph:
; SPLIT: Operations (5):
; SPLIT-NEXT: %l = call double @llvm.log.f64(double %T)
; SPLIT: [poseidon] After splitting, have 2 subgraphs in preprocess_tester
; SPLIT: [poseidon] joint sampling: 1 leaves, 1024 of 1024 draws inside the profiled input box
; SPLIT: [poseidon] joint sampling: 2 leaves, 1024 of 1024 draws inside the profiled input box
; SPLIT: Initial AccuracyCost: 1.601240e-25
; SPLIT: Initial AccuracyCost: 2.842171e-14

; HERBIE-CORE: (FPCore (v0) {{.*}} :pre (and (<= 2.50000000000000000e+02 v0 9.70000000000000000e+03)) :name "0" (pow v0 (fma (- (log v0)) 0.017899999999999999 -1.4847999999999999)))
; HERBIE-CORE: 1.466158e-25{{[[:space:]]+}}256{{[[:space:]]+}}{{.*}}(exp.f64 (*.f64 (fma.f64 (neg.f64 (log.f64 v0)) {{.*}} (log.f64 v0)))
; HERBIE-CORE: Applying solution for (pow v0 {{.*}} --(0)-> (pow.f64 v0 #s(literal -81/50 binary64))

; HERBIE-LABEL: define void @preprocess_tester(
; HERBIE: %l = tail call double @llvm.log.f64(double %T)
; HERBIE-NEXT: %herbie.pow = tail call double @pow(double %T, double -1.620000e+00)
