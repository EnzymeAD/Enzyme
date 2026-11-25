; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/pow_term/profiles \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-budget=-0.04 -poseidon-cache=%t/base -o /dev/null 2>&1 | FileCheck --check-prefix=BASE %s
; RUN: %{opt} -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-budget=-0.04 -poseidon-cache=%t/split -poseidon-subgraph-split-calls=pow -o /dev/null 2>&1 | FileCheck --check-prefix=SPLIT %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/pow_term %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=false -poseidon-cost-budget=-1e-9 -poseidon-subgraph-split-calls=pow -poseidon-cache=%t/herbie -o %t/herbie.ll 2>&1 | FileCheck --check-prefix=HERBIE-CORE %s
; RUN: FileCheck --check-prefix=HERBIE %s < %t/herbie.ll
; RUN: %{opt} -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-budget=-0.04 -poseidon-cache=%t/po -poseidon-output-profiles=%S/../../Inputs/pow_term/out0,%S/../../Inputs/pow_term/out1 -o /dev/null 2>&1 | FileCheck --check-prefix=PO %s
; RUN: not %{opt} -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-budget=-0.04 -poseidon-cache=%t/pomissing -poseidon-output-profiles=%S/../../Inputs/pow_term/out0,%t/none -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=PO-MISSING %s

; A rate term pow(T, fma(-log T, a, b)), scaled to about 1e-25, with log T also
; used by the second output.
;
; -poseidon-subgraph-split-calls=pow gives the pow call its own FP subgraph with the
; operations only it uses (the fma and the fneg; log T stays with its other
; user), whatever -poseidon-subgraph-split-uses/-poseidon-subgraph-split-ops say (unset
; here, which splits at no shared operation).
;
; The accuracy model draws T and y and evaluates log T from them, so every draw
; lands inside the profiled input box.
;
; The Herbie core of the pow subgraph spells l = log T as (log v0), and a
; (log v0) in a returned rewrite maps back to %l: the exact rewrite
; exp(fma(-log T, a, b) log T) is priced without a second log (Delta CompCost
; 256, not 768), and the rewrite pow(T, -1.62) is applied as llvm.pow.
;
; The accuracy weight of an instruction is floored at 1e-6 of the site's largest
; sensitivity |gradient x value|: the r1 chain, about 1e-29 of r2, is lifted (5
; of 8 instructions).
;
; -poseidon-output-profiles reads one profile per output, recorded with that
; output seeded alone, and weights each instruction by its largest gradient
; over that output's total sensitivity: no instruction is lifted.
;
; profiles: this module through the host profiler (opt
; -passes=poseidon,enzyme,poseidon-finalize -poseidon-profile-generate, linked with
; libposeidon_profile_shadows.a and libposeidon_profile.a), driven by
;   for (int i = 0; i < 64; ++i) {
;     double T = 250.0 + i * 150.0, y = 0.5 + (i % 7) * 0.25, o1, o2;
;     double d1 = only != 1, d2 = only != 0;
;     site(T, y, &o1, &d1, &o2, &d2);
;   }
; with only = -1; out0, out1: the same with only = 0 and only = 1, which seed
; one output's shadow.

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

; BASE: [poseidon] sensitivity floor raised the accuracy weight of 5 of 8 profiled instruction(s) in preprocess_tester to 1.000000e-06 x the site maximum sensitivity
; BASE: Found a subgraph with 8 operations and 2 inputs and 2 outputs
; BASE-NOT: Bottleneck:
; BASE: [poseidon] After splitting, have 1 subgraphs in preprocess_tester
; BASE: Initial AccuracyCost: 2.843141e-14

; SPLIT: Bottleneck:   %p = call double @llvm.pow.f64(double %T, double %e)
; SPLIT: Operations (3):
; SPLIT-NEXT: %p = call double @llvm.pow.f64
; SPLIT-NEXT: %e = call double @llvm.fma.f64
; SPLIT-NEXT: %nl = fneg double %l
; SPLIT: [poseidon] After splitting, have 2 subgraphs in preprocess_tester
; SPLIT: [poseidon] joint sampling: 1 leaves, 1024 of 1024 draws inside the profiled input box
; SPLIT: [poseidon] joint sampling: 2 leaves, 1024 of 1024 draws inside the profiled input box
; SPLIT: Initial AccuracyCost: 4.027094e-18
; SPLIT: Initial AccuracyCost: 2.842284e-14

; HERBIE-CORE: fake-herbie: {{.*}} --enable reduce:dag-cost
; HERBIE-CORE: (FPCore (v0) {{.*}} :pre (and (<= 2.50000000000000000e+02 v0 9.70000000000000000e+03)) :name "0" (pow v0 (fma (- (log v0)) 0.017899999999999999 -1.4847999999999999)))
; HERBIE-CORE: 3.687364e-18{{[[:space:]]+}}2.560000e+02{{[[:space:]]+}}{{.*}}(exp.f64 (*.f64 (fma.f64 (neg.f64 (log.f64 v0)) {{.*}} (log.f64 v0)))
; HERBIE-CORE: Applying solution for (pow v0 {{.*}} --(0)-> (pow.f64 v0 #s(literal -81/50 binary64))

; HERBIE-LABEL: define void @preprocess_tester(
; HERBIE: %l = tail call double @llvm.log.f64(double %T)
; HERBIE-NEXT: [[HERBIE_POW:%[^ ,()]+]] = tail call double @llvm.pow.f64(double %T, double -1.620000e+00)

; PO: [poseidon] preprocess_tester: accuracy weighted per output (2 output profiles)
; PO-NOT: sensitivity floor raised
; PO: Initial AccuracyCost: 2.600950e-17

; PO-MISSING: error: {{.*}}Poseidon: the profile directory {{.*}}none (-poseidon-output-profiles) does not exist; run the program built with -poseidon-profile-generate first, or pass the directory it wrote
