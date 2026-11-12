; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=SOLVED %s
; RUN: sed 's/"key": {/"key": {"zz extra": "1", "aa extra": "1", "mm extra": "1",/' %t/c/table.json > %t/table.json
; RUN: mv %t/table.json %t/c/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=EXTRA,SOLVED %s

; A cached table whose key has parts this compile does not names the first of
; them by name, whatever order the stored key object holds them in.

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %ys = call double @scale_a(double %y)
  %r1 = fmul double %k, %ys
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

declare double @scale_a(double)
declare double @scale_b(double)
declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; EXTRA: [poseidon] DP cache entry for preprocess_tester was solved for a different aa extra; re-solving.
; SOLVED: DP table for preprocess_tester cached to file.
